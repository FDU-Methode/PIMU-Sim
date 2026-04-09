# gen_resnet_block_dma6.py
# D = A * B + A
#
# Per output tile flow:
#   for each k-tile:
#       MVIN A(i,k) -> SPM bank0/1
#       MVIN B(k,j) -> SPM bank2/3
#       SA
#   MVOUT C(i,j) -> DRAM temp              (from SPM bank4/5, via DMA5/6)
#   MVIN residual A(i,j) -> ACC bank0      (via DMA3)
#   MVIN temp C(i,j) -> ACC bank1          (via DMA3)
#   MATADD(ACC0, ACC1) -> SPM bank6/7
#   MVOUT D(i,j) -> DRAM output            (from SPM bank6/7, via DMA5/6)

TILE = 32

# change as needed
M = 256
N = 256
K = 256

assert M % TILE == 0 and N % TILE == 0 and K % TILE == 0

MT = M // TILE
NT = N // TILE
KT = K // TILE

# DRAM strides in element count
LDA = K
LDB = N
LDC = N

DRAM_A_BASE   = 0x01000000
DRAM_B_BASE   = 0x02000000
DRAM_TMP_BASE = 0x03000000
DRAM_D_BASE   = 0x04000000

# SPM layout under current policy
A_BANK0 = 0x00000000   # bank0
A_BANK1 = 0x00040000   # bank1
B_BANK0 = 0x00080000   # bank2
B_BANK1 = 0x000C0000   # bank3
C_BANK0 = 0x00100000   # bank4
C_BANK1 = 0x00140000   # bank5
D_BANK0 = 0x00180000   # bank6
D_BANK1 = 0x001C0000   # bank7

# ACC data banks
ACC_BANK0 = 0x00000000
ACC_BANK1 = 0x00080000

TMP_TILE_BYTES = TILE * TILE * 4

def inst_hex(high, low):
    return f"{high:016x}{low:016x}"

def emit_dma(opcode, data_type, precision, is_quant, source, is_bias,
             col_num, row_num, sram_addr, dram_addr, sram_stride, dram_stride,
             qscale=0, qshift=0, qzp=0):
    low0 = (
        opcode
        | (data_type << 8)
        | (precision << 10)
        | (is_quant << 12)
        | (source << 13)
        | (is_bias << 14)
        | (col_num << 32)
        | (row_num << 48)
    )
    high0 = sram_addr | (dram_addr << 32)

    low1 = (
        sram_stride
        | (dram_stride << 16)
        | (qscale << 32)
        | (qshift << 48)
    )
    high1 = qzp

    return [inst_hex(high0, low0), inst_hex(high1, low1)]

def emit_sa(config1, config2,
            bias_w, bias_h, bias_stride, output_stride,
            biaspsum_addr, output_addr,
            a_addr, a_col_num, a_row_num, a_stride,
            b_addr, b_col_num, b_row_num, b_stride,
            input_a_zp=0, input_b_zp=0, output_scale=1, output_shift=0, output_zp=0):
    low0 = (
        0x03
        | (config1 << 8)
        | (input_a_zp << 32)
        | (input_b_zp << 48)
    )
    high0 = output_scale | (output_shift << 16) | (output_zp << 32)

    low1 = (
        (config2 & 0x1FFF)
        | (bias_w << 16)
        | (bias_h << 24)
        | (bias_stride << 32)
        | (output_stride << 48)
    )
    high1 = biaspsum_addr | (output_addr << 32)

    low2 = (
        a_addr
        | (a_col_num << 32)
        | (a_row_num << 43)
        | (a_stride << 48)
    )
    high2 = (
        b_addr
        | (b_col_num << 32)
        | (b_row_num << 37)
        | (b_stride << 48)
    )

    return [inst_hex(high0, low0), inst_hex(high1, low1), inst_hex(high2, low2)]

def emit_matadd(a_addr, b_addr, out_addr, col_num, row_num,
                output_scale=1, output_shift=0, output_zp=0):
    low0 = 0x04 | (col_num << 8) | (row_num << 16) | (a_addr << 32)
    high0 = b_addr | (out_addr << 32)

    low1 = output_scale | (output_shift << 16) | (output_zp << 32)
    high1 = 0

    low2 = 0
    high2 = 0

    return [inst_hex(high0, low0), inst_hex(high1, low1), inst_hex(high2, low2)]

def sa_config2(acc_dest, a_keep=0, b_keep=0, is_acc=0, is_bias=0, bias_keep=0):
    cfg = 0
    cfg |= (acc_dest  & 0x1) << 3
    cfg |= (a_keep    & 0x1) << 4
    cfg |= (b_keep    & 0x1) << 5
    cfg |= (is_acc    & 0x1) << 6
    cfg |= (is_bias   & 0x1) << 11
    cfg |= (bias_keep & 0x1) << 12
    return cfg

def dram_addr_a(i_tile, k_tile):
    return DRAM_A_BASE + (i_tile * TILE) * LDA + (k_tile * TILE)

def dram_addr_b(k_tile, j_tile):
    return DRAM_B_BASE + (k_tile * TILE) * LDB + (j_tile * TILE)

def dram_addr_residual_a(i_tile, j_tile):
    return DRAM_A_BASE + (i_tile * TILE) * LDA + (j_tile * TILE)

def dram_addr_tmp(i_tile, j_tile):
    tile_id = i_tile * NT + j_tile
    return DRAM_TMP_BASE + tile_id * TMP_TILE_BYTES

def dram_addr_d(i_tile, j_tile):
    return DRAM_D_BASE + (i_tile * TILE) * LDC + (j_tile * TILE)

def generate():
    lines = []
    lines.append("# ResNet block under fixed-bank / fixed-DMA policy")
    lines.append("# bank0-1:A bank2-3:B bank4-5:C bank6-7:D")
    lines.append("# D = A * B + A")

    config1_gemm = 0
    cfg_sa_first = sa_config2(acc_dest=1, is_acc=0, is_bias=0)
    cfg_sa_mid   = sa_config2(acc_dest=1, is_acc=1, is_bias=0)
    cfg_sa_last  = sa_config2(acc_dest=0, is_acc=1, is_bias=0)

    for i in range(MT):
        for j in range(NT):
            lines.append(f"# ---- Output Tile ({i}, {j}) ----")

            c_bank = C_BANK0 if ((i + j) & 1) == 0 else C_BANK1
            d_bank = D_BANK0 if ((i + j) & 1) == 0 else D_BANK1

            for k in range(KT):
                a_bank = A_BANK0 if (k & 1) == 0 else A_BANK1
                b_bank = B_BANK0 if (k & 1) == 0 else B_BANK1

                lines.append(f"# MVIN A tile ({i},{k}) -> bank{'0' if (k & 1) == 0 else '1'}")
                lines.extend(
                    emit_dma(
                        opcode=0x01,
                        data_type=0,
                        precision=1,
                        is_quant=0,
                        source=0,   # dest = SPM
                        is_bias=0,
                        col_num=TILE - 1,
                        row_num=TILE - 1,
                        sram_addr=a_bank,
                        dram_addr=dram_addr_a(i, k),
                        sram_stride=TILE,
                        dram_stride=LDA,
                    )
                )

                lines.append(f"# MVIN B tile ({k},{j}) -> bank{'2' if (k & 1) == 0 else '3'}")
                lines.extend(
                    emit_dma(
                        opcode=0x01,
                        data_type=0,
                        precision=1,
                        is_quant=0,
                        source=0,   # dest = SPM
                        is_bias=0,
                        col_num=TILE - 1,
                        row_num=TILE - 1,
                        sram_addr=b_bank,
                        dram_addr=dram_addr_b(k, j),
                        sram_stride=TILE,
                        dram_stride=LDB,
                    )
                )

                if k == 0:
                    cfg2 = cfg_sa_first
                    biaspsum_addr = 0
                    output_addr = ACC_BANK1
                elif k == KT - 1:
                    cfg2 = cfg_sa_last
                    biaspsum_addr = ACC_BANK1
                    output_addr = c_bank
                else:
                    cfg2 = cfg_sa_mid
                    biaspsum_addr = ACC_BANK1
                    output_addr = ACC_BANK1

                lines.append(f"# SA tile ({i},{j},{k})")
                lines.extend(
                    emit_sa(
                        config1=config1_gemm,
                        config2=cfg2,
                        bias_w=0,
                        bias_h=0,
                        bias_stride=TILE,
                        output_stride=TILE,
                        biaspsum_addr=biaspsum_addr,
                        output_addr=output_addr,
                        a_addr=a_bank,
                        a_col_num=TILE - 1,
                        a_row_num=TILE - 1,
                        a_stride=TILE,
                        b_addr=b_bank,
                        b_col_num=TILE - 1,
                        b_row_num=TILE - 1,
                        b_stride=TILE,
                    )
                )

            lines.append(f"# MVOUT C tile ({i},{j}) -> DDR temp")
            lines.extend(
                emit_dma(
                    opcode=0x02,
                    data_type=0,
                    precision=1,
                    is_quant=0,
                    source=0,   # source = SPM
                    is_bias=0,
                    col_num=TILE - 1,
                    row_num=TILE - 1,
                    sram_addr=c_bank,
                    dram_addr=dram_addr_tmp(i, j),
                    sram_stride=TILE,
                    dram_stride=TILE,
                )
            )

            lines.append(f"# MVIN residual A tile ({i},{j}) -> ACC bank0")
            lines.extend(
                emit_dma(
                    opcode=0x01,
                    data_type=2,
                    precision=1,
                    is_quant=0,
                    source=1,   # dest = ACC
                    is_bias=0,
                    col_num=TILE - 1,
                    row_num=TILE - 1,
                    sram_addr=ACC_BANK0,
                    dram_addr=dram_addr_residual_a(i, j),
                    sram_stride=TILE,
                    dram_stride=LDA,
                )
            )

            lines.append(f"# MVIN temp C tile ({i},{j}) -> ACC bank1")
            lines.extend(
                emit_dma(
                    opcode=0x01,
                    data_type=2,
                    precision=1,
                    is_quant=0,
                    source=1,   # dest = ACC
                    is_bias=0,
                    col_num=TILE - 1,
                    row_num=TILE - 1,
                    sram_addr=ACC_BANK1,
                    dram_addr=dram_addr_tmp(i, j),
                    sram_stride=TILE,
                    dram_stride=TILE,
                )
            )

            lines.append(f"# MATADD tile ({i},{j}) -> D bank")
            lines.extend(
                emit_matadd(
                    a_addr=ACC_BANK0,
                    b_addr=ACC_BANK1,
                    out_addr=d_bank,
                    col_num=TILE - 1,
                    row_num=TILE - 1,
                    output_scale=1,
                    output_shift=0,
                    output_zp=0,
                )
            )

            lines.append(f"# MVOUT D tile ({i},{j}) -> DRAM output")
            lines.extend(
                emit_dma(
                    opcode=0x02,
                    data_type=0,
                    precision=1,
                    is_quant=0,
                    source=0,   # source = SPM
                    is_bias=0,
                    col_num=TILE - 1,
                    row_num=TILE - 1,
                    sram_addr=d_bank,
                    dram_addr=dram_addr_d(i, j),
                    sram_stride=TILE,
                    dram_stride=LDC,
                )
            )

    return lines

if __name__ == "__main__":
    out_file = "./benchmarks/tb_test/tb_resnet_block.txt"
    with open(out_file, "w") as f:
        for line in generate():
            f.write(line + "\n")
    print(f"generated {out_file}")
