TILE = 32
M = 1024
N = 1024
K = 1024

assert M % TILE == 0 and N % TILE == 0 and K % TILE == 0

MT = M // TILE
NT = N // TILE
KT = K // TILE

# row-major, int8 layout in DRAM
LDA = 1024
LDB = 1024
LDC = 1024

DRAM_A_BASE = 0x01000000
DRAM_B_BASE = 0x02000000
DRAM_C_BASE = 0x03000000

# current cfg:
# sram_bank_bytes = 0x40000
# acc_bytes = 0x100000, acc_bank_num = 2 => acc bank boundary = 0x80000

# SPM banks
A_BANK0   = 0x00000000   # bank0
A_BANK1   = 0x00040000   # bank1
B_BANK0   = 0x00080000   # bank2
B_BANK1   = 0x000C0000   # bank3
OUT_BANK0 = 0x00100000   # bank4
OUT_BANK1 = 0x00140000   # bank5

# ACC data banks
A_ACC = 0x00000000       # acc data bank0, shortcut
Y_ACC = 0x00080000       # acc data bank1, GEMM partial/final sum

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

def emit_sa(config1, input_a_zp, input_b_zp, output_scale, output_shift, output_zp,
            config2, bias_w, bias_h, bias_stride, output_stride, biaspsum_addr, output_addr,
            a_addr, a_col_num, a_row_num, a_stride,
            b_addr, b_col_num, b_row_num, b_stride):
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

def dram_addr_a(i_tile, k_tile):
    return DRAM_A_BASE + (i_tile * TILE) * LDA + (k_tile * TILE)

def dram_addr_b(k_tile, j_tile):
    return DRAM_B_BASE + (k_tile * TILE) * LDB + (j_tile * TILE)

def dram_addr_residual_a(i_tile, j_tile):
    return DRAM_A_BASE + (i_tile * TILE) * LDA + (j_tile * TILE)

def dram_addr_c(i_tile, j_tile):
    return DRAM_C_BASE + (i_tile * TILE) * LDC + (j_tile * TILE)

def generate():
    lines = []
    lines.append("# ResNet-like block: Y = A * B + A")
    lines.append(f"# shape: A={M}x{K}, B={K}x{N}, Y={M}x{N}")
    lines.append(f"# tile: {TILE}x{TILE}")

    # GEMM:
    # config2 bit usage in current code:
    # bit[3]  acc_dest
    # bit[6]  is_acc
    # bit[11] is_bias
    # bit[12] bias_valid
    #
    # first k-tile: acc_dest=1, is_acc=0
    # later k-tile: acc_dest=1, is_acc=1
    config1_gemm = 0
    config2_first = (1 << 3)              # acc_dest=1
    config2_accum = (1 << 3) | (1 << 6)   # acc_dest=1, is_acc=1

    for i in range(MT):
        for j in range(NT):
            lines.append(f"# ---- Output Tile ({i}, {j}) ----")

            # shortcut A(i,j) -> ACC data bank0
            lines.append(f"# MVIN shortcut A({i},{j}) -> ACC data bank0")
            lines.extend(
                emit_dma(
                    opcode=0x01,
                    data_type=2,      # int32 -> ACC data
                    precision=1,
                    is_quant=0,
                    source=1,
                    is_bias=0,
                    col_num=TILE - 1,
                    row_num=TILE - 1,
                    sram_addr=A_ACC,
                    dram_addr=dram_addr_residual_a(i, j),
                    sram_stride=TILE,
                    dram_stride=LDA,
                )
            )

            for k in range(KT):
                a_bank = A_BANK0 if (k & 1) == 0 else A_BANK1
                b_bank = B_BANK0 if (k & 1) == 0 else B_BANK1

                lines.append(f"# MVIN A tile ({i},{k}) -> SPM")
                lines.extend(
                    emit_dma(
                        opcode=0x01,
                        data_type=0,
                        precision=1,
                        is_quant=0,
                        source=0,
                        is_bias=0,
                        col_num=TILE - 1,
                        row_num=TILE - 1,
                        sram_addr=a_bank,
                        dram_addr=dram_addr_a(i, k),
                        sram_stride=TILE,
                        dram_stride=LDA,
                    )
                )

                lines.append(f"# MVIN B tile ({k},{j}) -> SPM")
                lines.extend(
                    emit_dma(
                        opcode=0x01,
                        data_type=0,
                        precision=1,
                        is_quant=0,
                        source=0,
                        is_bias=0,
                        col_num=TILE - 1,
                        row_num=TILE - 1,
                        sram_addr=b_bank,
                        dram_addr=dram_addr_b(k, j),
                        sram_stride=TILE,
                        dram_stride=LDB,
                    )
                )

                first_k = (k == 0)
                config2 = config2_first if first_k else config2_accum
                biaspsum_addr = 0 if first_k else Y_ACC

                lines.append(f"# SA tile ({i},{j},{k}) -> ACC data bank1")
                lines.extend(
                    emit_sa(
                        config1=0,  # GEMM
                        input_a_zp=0,
                        input_b_zp=0,
                        output_scale=1,
                        output_shift=0,
                        output_zp=0,
                        config2=config2,
                        bias_w=0,
                        bias_h=0,
                        bias_stride=TILE,
                        output_stride=TILE,
                        biaspsum_addr=biaspsum_addr,
                        output_addr=Y_ACC,
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

            out_bank = OUT_BANK0 if ((i + j) & 1) == 0 else OUT_BANK1

            lines.append(f"# MATADD residual tile ({i},{j})")
            lines.extend(
                emit_matadd(
                    a_addr=A_ACC,
                    b_addr=Y_ACC,
                    out_addr=out_bank,
                    col_num=TILE - 1,
                    row_num=TILE - 1,
                    output_scale=1,
                    output_shift=0,
                    output_zp=0,
                )
            )

            lines.append(f"# MVOUT output tile ({i},{j})")
            lines.extend(
                emit_dma(
                    opcode=0x02,
                    data_type=2,
                    precision=1,
                    is_quant=0,
                    source=0,
                    is_bias=0,
                    col_num=TILE - 1,
                    row_num=TILE - 1,
                    sram_addr=out_bank,
                    dram_addr=dram_addr_c(i, j),
                    sram_stride=TILE,
                    dram_stride=LDC,
                )
            )


    return lines

if __name__ == "__main__":
    with open("./benchmarks/tb_test/tb_resnet_block_1024.txt", "w") as f:
        for line in generate():
            f.write(line + "\n")
    print("generated tb_resnet_block_1024.txt")
