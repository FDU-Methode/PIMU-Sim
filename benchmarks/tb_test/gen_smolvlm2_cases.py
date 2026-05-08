#!/usr/bin/env python3
from pathlib import Path

TILE = 32
PREFILL_TOKENS = 32
DECODE_TOKENS = 1

HIDDEN = 960
KV_DIM = 320
HEAD_DIM = 64
NUM_Q_HEADS = 15
NUM_KV_HEADS = 5
Q_PER_KV_HEAD = NUM_Q_HEADS // NUM_KV_HEADS
INTERMEDIATE = 2560

ACT_CHUNK = 1024

assert NUM_Q_HEADS % NUM_KV_HEADS == 0
assert HIDDEN == NUM_Q_HEADS * HEAD_DIM
assert KV_DIM == NUM_KV_HEADS * HEAD_DIM

# Prefill fills slots [0, 31]. Decode appends slot 32 and attends over all
# 33 cache positions. The K cache is stored transposed with this row stride.
KV_CACHE_STRIDE = PREFILL_TOKENS + DECODE_TOKENS

DRAM_PREFILL_INPUT = 0x01000000
DRAM_DECODE_INPUT = 0x01100000

DRAM_W_Q = 0x02000000
DRAM_W_K = 0x03000000
DRAM_W_V = 0x04000000
DRAM_W_O = 0x05000000
DRAM_W_FFN_UP = 0x06000000
DRAM_W_FFN_DOWN = 0x07000000

DRAM_LN1 = 0x08000000
DRAM_PREFILL_Q = 0x08100000
DRAM_K_CACHE_T = 0x09000000
DRAM_V_CACHE = 0x0A000000
DRAM_SCORE = 0x0B000000
DRAM_PROB = 0x0C000000
DRAM_CONTEXT = 0x0D000000
DRAM_ATTN_OUT = 0x0E000000
DRAM_RESID1 = 0x0F000000
DRAM_LN2 = 0x10000000
DRAM_FFN_UP = 0x11000000
DRAM_FFN_ACT = 0x12000000
DRAM_FFN_DOWN = 0x13000000
DRAM_PREFILL_OUT = 0x14000000

DRAM_DECODE_LN1 = 0x15000000
DRAM_DECODE_Q = 0x15100000
DRAM_DECODE_K_CACHE_T = DRAM_K_CACHE_T + PREFILL_TOKENS
DRAM_DECODE_V_CACHE = DRAM_V_CACHE + PREFILL_TOKENS * KV_DIM
DRAM_DECODE_ATTN_OUT = 0x18000000
DRAM_DECODE_RESID1 = 0x19000000
DRAM_DECODE_LN2 = 0x1A000000
DRAM_DECODE_FFN_UP = 0x1B000000
DRAM_DECODE_FFN_ACT = 0x1C000000
DRAM_DECODE_FFN_DOWN = 0x1D000000
DRAM_DECODE_OUT = 0x1E000000

A_BUF = 0x00000000
B_BUF = 0x00001000
C_BUF = 0x00002000
T_BUF = 0x00003000
SPM_ACT_IN = 0x00010000
SPM_ACT_OUT = 0x00018000

ACC_A_BUF = 0x00000000
ACC_PSUM_BUF = 0x00001000


def word64(value: int) -> str:
    hi = (value >> 32) & 0xFFFFFFFF
    lo = value & 0xFFFFFFFF
    return f"64'h{hi:08x}_{lo:08x}"


def reg32(value: int) -> str:
    return f"32'h{value:02x}"


def emit_write(op_name: str, reg_addr: int, value: int) -> str:
    return f"{op_name} {reg32(reg_addr)} {word64(value)}"


def emit_start(bit_value: int):
    return [
        emit_write("ENR", 0xB0, bit_value),
        f"# IAR {reg32(0xB8)} {word64(bit_value)}",
    ]


ADDR = {
    "MVIN_RS1": 0x00,
    "MVIN_RS2": 0x08,
    "MVOUT_RS1": 0x10,
    "MVOUT_RS2": 0x18,
    "CFG_MVIN_RS1": 0x20,
    "CFG_MVIN_RS2": 0x28,
    "CFG_MVOUT_RS1": 0x30,
    "CFG_MVOUT_RS2": 0x38,
    "CFG_COMPUTE_RS1": 0x40,
    "CFG_COMPUTE_RS2": 0x48,
    "CFG_SFU_RS1": 0x50,
    "CFG_SFU_RS2": 0x58,
    "CFG_ACC_RS1": 0x60,
    "CFG_ACC_RS2": 0x68,
    "COMPUTE_SA_RS1": 0x70,
    "COMPUTE_SA_RS2": 0x78,
    "COMPUTE_MATADD_RS1": 0x80,
    "COMPUTE_MATADD_RS2": 0x88,
    "COMPUTE_SFU_RS1": 0xA0,
    "COMPUTE_SFU_RS2": 0xA8,
}


def encode_mvin_cfg(dram_stride: int, sram_stride: int, to_acc: bool) -> int:
    flags = 0
    if to_acc:
        flags |= 1 << 1
        flags |= 1 << 5
    flags |= 1 << 2
    return (dram_stride << 48) | (sram_stride << 32) | flags


def encode_mvout_cfg(dram_stride: int, sram_stride: int, from_acc: bool) -> int:
    flags = 0
    if from_acc:
        flags |= 1 << 1
        flags |= 1 << 5
    flags |= 1 << 2
    return (dram_stride << 48) | (sram_stride << 32) | flags


def emit_mvin(dram_addr: int, local_addr: int, row_num: int, col_num: int,
              dram_stride: int, sram_stride: int, to_acc: bool):
    return [
        emit_write("mvin", ADDR["MVIN_RS1"], ((row_num - 1) << 32) | dram_addr),
        emit_write("mvin", ADDR["MVIN_RS2"], ((col_num - 1) << 32) | local_addr),
        emit_write("config_mvin", ADDR["CFG_MVIN_RS1"], encode_mvin_cfg(dram_stride, sram_stride, to_acc)),
        emit_write("config_mvin", ADDR["CFG_MVIN_RS2"], 0x0000000100000000),
        *emit_start(0x1),
    ]


def emit_mvout(local_addr: int, dram_addr: int, row_num: int, col_num: int,
               dram_stride: int, sram_stride: int, from_acc: bool):
    return [
        emit_write("mvout", ADDR["MVOUT_RS1"], ((row_num - 1) << 32) | local_addr),
        emit_write("mvout", ADDR["MVOUT_RS2"], ((col_num - 1) << 32) | dram_addr),
        emit_write("config_mvout", ADDR["CFG_MVOUT_RS1"], encode_mvout_cfg(dram_stride, sram_stride, from_acc)),
        emit_write("config_mvout", ADDR["CFG_MVOUT_RS2"], 0x0000000100000000),
        *emit_start(0x2),
    ]


def emit_sa(config2: int, bias_w: int, bias_h: int, bias_stride: int, output_stride: int,
            biaspsum_addr: int, output_addr: int, a_addr: int, a_col_num: int, a_row_num: int, a_stride: int,
            b_addr: int, b_col_num: int, b_row_num: int, b_stride: int,
            input_a_zp: int = 0, input_b_zp: int = 0, output_scale: int = 1,
            output_shift: int = 0, output_zp: int = 0):
    acc_dest = (config2 >> 3) & 0x1
    is_acc = (config2 >> 6) & 0x1
    cfg_compute_rs1 = 0x00000001 | (acc_dest << 27)
    cfg_acc_rs1 = biaspsum_addr | (bias_stride << 32)
    cfg_acc_rs2 = output_addr | (output_stride << 32) | (is_acc << 48)
    compute_rs1 = (
        a_addr
        | ((a_col_num - 1) << 32)
        | ((a_row_num - 1) << 43)
        | (a_stride << 48)
    )
    compute_rs2 = (
        b_addr
        | ((b_col_num - 1) << 32)
        | ((b_row_num - 1) << 37)
        | (b_stride << 48)
    )
    return [
        emit_write("config_compute", ADDR["CFG_COMPUTE_RS1"], cfg_compute_rs1),
        emit_write("config_compute", ADDR["CFG_COMPUTE_RS2"], 0x0000000100000000),
        emit_write("config_accumulate", ADDR["CFG_ACC_RS1"], cfg_acc_rs1),
        emit_write("config_accumulate", ADDR["CFG_ACC_RS2"], cfg_acc_rs2),
        emit_write("compute_sa", ADDR["COMPUTE_SA_RS1"], compute_rs1),
        emit_write("compute_sa", ADDR["COMPUTE_SA_RS2"], compute_rs2),
        *emit_start(0x4),
    ]


def emit_matadd(a_addr: int, b_addr: int, out_addr: int, row_num: int, col_num: int,
                output_scale: int = 1, output_shift: int = 0, output_zp: int = 0):
    rs1 = (b_addr << 32) | a_addr
    rs2 = out_addr | ((col_num - 1) << 32) | ((row_num - 1) << 40)
    return [
        emit_write("config_compute", ADDR["CFG_COMPUTE_RS1"], 0x0000000000000001),
        emit_write("config_compute", ADDR["CFG_COMPUTE_RS2"], 0x0000000100000000),
        emit_write("compute_matadd", ADDR["COMPUTE_MATADD_RS1"], rs1),
        emit_write("compute_matadd", ADDR["COMPUTE_MATADD_RS2"], rs2),
        *emit_start(0x8),
    ]


def emit_sfu(optype: int, input_addr: int, output_addr: int, row_num: int, col_num: int, is_quant: int = 0):
    cfg_rs1 = optype | (is_quant << 8)
    compute_rs1 = input_addr | ((col_num - 1) << 32) | ((row_num - 1) << 48)
    return [
        emit_write("config_sfu", ADDR["CFG_SFU_RS1"], cfg_rs1),
        emit_write("config_sfu", ADDR["CFG_SFU_RS2"], 0x0000000100000001),
        emit_write("compute_sfu", ADDR["COMPUTE_SFU_RS1"], compute_rs1),
        emit_write("compute_sfu", ADDR["COMPUTE_SFU_RS2"], output_addr),
        *emit_start(0x20),
    ]


def inst_hex(high: int, low: int) -> str:
    return f"{high:016x}{low:016x}"


def tile_ranges(size: int):
    return range(0, size, TILE)


def tile_size(size: int, offset: int) -> int:
    return min(TILE, size - offset)


def dram_addr(base: int, row_offset: int, col_offset: int, stride: int) -> int:
    return base + row_offset * stride + col_offset


def emit_gemm_matrix(lines, name: str, a_base: int, a_stride: int, b_base: int, b_stride: int,
                     out_base: int, out_stride: int, rows: int, cols: int, depth: int,
                     transpose_out: bool = False, transpose_stride: int = 0):
    lines.append(f"# {name}")
    for i in tile_ranges(rows):
        row_t = tile_size(rows, i)
        for j in tile_ranges(cols):
            col_t = tile_size(cols, j)
            lines.append(f"# GEMM tile ({i // TILE}, {j // TILE})")

            for k in tile_ranges(depth):
                k_t = tile_size(depth, k)
                a_bank = A_BUF
                b_bank = B_BUF

                lines.extend(
                    emit_mvin(
                        dram_addr(a_base, i, k, a_stride),
                        a_bank,
                        row_t,
                        k_t,
                        a_stride,
                        k_t,
                        False,
                    )
                )
                lines.extend(
                    emit_mvin(
                        dram_addr(b_base, k, j, b_stride),
                        b_bank,
                        k_t,
                        col_t,
                        b_stride,
                        col_t,
                        False,
                    )
                )

                if depth <= TILE:
                    cfg2 = 0
                    biaspsum_addr = 0
                    output_addr = C_BUF
                elif k == 0:
                    cfg2 = 1 << 3
                    biaspsum_addr = 0
                    output_addr = ACC_PSUM_BUF
                elif k + TILE < depth:
                    cfg2 = (1 << 3) | (1 << 6)
                    biaspsum_addr = ACC_PSUM_BUF
                    output_addr = ACC_PSUM_BUF
                else:
                    cfg2 = (1 << 6)
                    biaspsum_addr = ACC_PSUM_BUF
                    output_addr = C_BUF

                lines.append(f"# SA tile ({i // TILE}, {j // TILE}, {k // TILE})")
                lines.extend(
                    emit_sa(
                        config2=cfg2,
                        bias_w=0,
                        bias_h=0,
                        bias_stride=col_t,
                        output_stride=col_t,
                        biaspsum_addr=biaspsum_addr,
                        output_addr=output_addr,
                        a_addr=a_bank,
                        a_col_num=k_t,
                        a_row_num=row_t,
                        a_stride=k_t,
                        b_addr=b_bank,
                        b_col_num=col_t,
                        b_row_num=k_t,
                        b_stride=col_t,
                    )
                )

            if transpose_out:
                lines.append(f"# Transpose tile ({i // TILE}, {j // TILE})")
                lines.extend(emit_sfu(8, C_BUF, T_BUF, row_t, col_t))
                lines.extend(
                    emit_mvout(
                        T_BUF,
                        dram_addr(out_base, j, i, transpose_stride),
                        col_t,
                        row_t,
                        transpose_stride,
                        row_t,
                        False,
                    )
                )
            else:
                lines.append(f"# MVOUT tile ({i // TILE}, {j // TILE})")
                lines.extend(
                    emit_mvout(
                        C_BUF,
                        dram_addr(out_base, i, j, out_stride),
                        row_t,
                        col_t,
                        out_stride,
                        col_t,
                        False,
                    )
                )


def emit_layernorm(lines, name: str, src_base: int, src_stride: int, dst_base: int, dst_stride: int,
                   rows: int, cols: int):
    lines.append(f"# {name}")
    lines.extend(emit_mvin(src_base, SPM_ACT_IN, rows, cols, src_stride, cols, False))
    lines.extend(emit_sfu(2, SPM_ACT_IN, SPM_ACT_OUT, rows, cols))
    lines.extend(emit_mvout(SPM_ACT_OUT, dst_base, rows, cols, dst_stride, cols, False))


def emit_softmax(lines, name: str, src_base: int, src_stride: int, dst_base: int, dst_stride: int,
                 rows: int, cols: int):
    lines.append(f"# {name}")
    lines.extend(emit_mvin(src_base, SPM_ACT_IN, rows, cols, src_stride, cols, False))
    lines.extend(emit_sfu(0, SPM_ACT_IN, SPM_ACT_OUT, rows, cols))
    lines.extend(emit_mvout(SPM_ACT_OUT, dst_base, rows, cols, dst_stride, cols, False))


def emit_activation(lines, name: str, src_base: int, src_stride: int, dst_base: int, dst_stride: int,
                    rows: int, cols: int):
    lines.append(f"# {name}")
    for col_off in range(0, cols, ACT_CHUNK):
        chunk = min(ACT_CHUNK, cols - col_off)
        lines.append(f"# activation chunk {col_off // ACT_CHUNK}")
        lines.extend(
            emit_mvin(
                src_base + col_off,
                SPM_ACT_IN,
                rows,
                chunk,
                src_stride,
                chunk,
                False,
            )
        )
        lines.extend(emit_sfu(1, SPM_ACT_IN, SPM_ACT_OUT, rows, chunk))
        lines.extend(
            emit_mvout(
                SPM_ACT_OUT,
                dst_base + col_off,
                rows,
                chunk,
                dst_stride,
                chunk,
                False,
            )
        )


def emit_residual_add(lines, name: str, a_base: int, a_stride: int, b_base: int, b_stride: int,
                      out_base: int, out_stride: int, rows: int, cols: int):
    lines.append(f"# {name}")
    for i in tile_ranges(rows):
        row_t = tile_size(rows, i)
        for j in tile_ranges(cols):
            col_t = tile_size(cols, j)
            lines.append(f"# MATADD tile ({i // TILE}, {j // TILE})")
            lines.extend(
                emit_mvin(
                    dram_addr(a_base, i, j, a_stride),
                    ACC_A_BUF,
                    row_t,
                    col_t,
                    a_stride,
                    col_t,
                    True,
                )
            )
            lines.extend(
                emit_mvin(
                    dram_addr(b_base, i, j, b_stride),
                    ACC_PSUM_BUF,
                    row_t,
                    col_t,
                    b_stride,
                    col_t,
                    True,
                )
            )
            lines.extend(emit_matadd(ACC_A_BUF, ACC_PSUM_BUF, C_BUF, row_t, col_t))
            lines.extend(
                emit_mvout(
                    C_BUF,
                    dram_addr(out_base, i, j, out_stride),
                    row_t,
                    col_t,
                    out_stride,
                    col_t,
                    False,
                )
            )


def emit_attention(lines, prefix: str, q_base_root: int, k_cache_root: int, v_cache_root: int,
                   rows: int, cache_cols: int, cache_stride: int):
    for q_head in range(NUM_Q_HEADS):
        kv_head = q_head // Q_PER_KV_HEAD
        q_base = q_base_root + q_head * HEAD_DIM
        k_base = k_cache_root + kv_head * HEAD_DIM * cache_stride
        v_base = v_cache_root + kv_head * HEAD_DIM
        context_base = DRAM_CONTEXT + q_head * HEAD_DIM

        emit_gemm_matrix(
            lines,
            f"{prefix} attention score head {q_head}",
            q_base,
            HIDDEN,
            k_base,
            cache_stride,
            DRAM_SCORE,
            cache_cols,
            rows,
            cache_cols,
            HEAD_DIM,
        )

        emit_softmax(
            lines,
            f"{prefix} softmax head {q_head}",
            DRAM_SCORE,
            cache_cols,
            DRAM_PROB,
            cache_cols,
            rows,
            cache_cols,
        )

        emit_gemm_matrix(
            lines,
            f"{prefix} attention context head {q_head}",
            DRAM_PROB,
            cache_cols,
            v_base,
            KV_DIM,
            context_base,
            HIDDEN,
            rows,
            HEAD_DIM,
            cache_cols,
        )


def build_prefill():
    lines = [
        "MER 32'hC0 64'h01",
        "IER 32'hC8 64'h3F",
        "",
        "# SmolVLM2-500M-Video-Instruct prefill",
        "# block = RMSNorm -> Q/K/V -> attention -> O -> residual -> FFN -> residual",
        "# SmolVLM2 uses a gated SiLU FFN; the simulator exercises the supported activation path.",
        "",
    ]

    emit_layernorm(
        lines,
        "Prefill layernorm 1",
        DRAM_PREFILL_INPUT,
        HIDDEN,
        DRAM_LN1,
        HIDDEN,
        PREFILL_TOKENS,
        HIDDEN,
    )

    emit_gemm_matrix(
        lines,
        "Prefill Q projection",
        DRAM_LN1,
        HIDDEN,
        DRAM_W_Q,
        HIDDEN,
        DRAM_PREFILL_Q,
        HIDDEN,
        PREFILL_TOKENS,
        HIDDEN,
        HIDDEN,
    )

    emit_gemm_matrix(
        lines,
        "Prefill K projection",
        DRAM_LN1,
        HIDDEN,
        DRAM_W_K,
        KV_DIM,
        DRAM_K_CACHE_T,
        KV_CACHE_STRIDE,
        PREFILL_TOKENS,
        KV_DIM,
        HIDDEN,
        transpose_out=True,
        transpose_stride=KV_CACHE_STRIDE,
    )

    emit_gemm_matrix(
        lines,
        "Prefill V projection",
        DRAM_LN1,
        HIDDEN,
        DRAM_W_V,
        KV_DIM,
        DRAM_V_CACHE,
        KV_DIM,
        PREFILL_TOKENS,
        KV_DIM,
        HIDDEN,
    )

    emit_attention(lines, "Prefill", DRAM_PREFILL_Q, DRAM_K_CACHE_T, DRAM_V_CACHE,
                   PREFILL_TOKENS, PREFILL_TOKENS, KV_CACHE_STRIDE)

    emit_gemm_matrix(
        lines,
        "Prefill output projection",
        DRAM_CONTEXT,
        HIDDEN,
        DRAM_W_O,
        HIDDEN,
        DRAM_ATTN_OUT,
        HIDDEN,
        PREFILL_TOKENS,
        HIDDEN,
        HIDDEN,
    )

    emit_residual_add(
        lines,
        "Prefill residual 1",
        DRAM_PREFILL_INPUT,
        HIDDEN,
        DRAM_ATTN_OUT,
        HIDDEN,
        DRAM_RESID1,
        HIDDEN,
        PREFILL_TOKENS,
        HIDDEN,
    )

    emit_layernorm(
        lines,
        "Prefill layernorm 2",
        DRAM_RESID1,
        HIDDEN,
        DRAM_LN2,
        HIDDEN,
        PREFILL_TOKENS,
        HIDDEN,
    )

    emit_gemm_matrix(
        lines,
        "Prefill FFN up projection",
        DRAM_LN2,
        HIDDEN,
        DRAM_W_FFN_UP,
        INTERMEDIATE,
        DRAM_FFN_UP,
        INTERMEDIATE,
        PREFILL_TOKENS,
        INTERMEDIATE,
        HIDDEN,
    )

    emit_activation(
        lines,
        "Prefill FFN activation",
        DRAM_FFN_UP,
        INTERMEDIATE,
        DRAM_FFN_ACT,
        INTERMEDIATE,
        PREFILL_TOKENS,
        INTERMEDIATE,
    )

    emit_gemm_matrix(
        lines,
        "Prefill FFN down projection",
        DRAM_FFN_ACT,
        INTERMEDIATE,
        DRAM_W_FFN_DOWN,
        HIDDEN,
        DRAM_FFN_DOWN,
        HIDDEN,
        PREFILL_TOKENS,
        HIDDEN,
        INTERMEDIATE,
    )

    emit_residual_add(
        lines,
        "Prefill residual 2",
        DRAM_RESID1,
        HIDDEN,
        DRAM_FFN_DOWN,
        HIDDEN,
        DRAM_PREFILL_OUT,
        HIDDEN,
        PREFILL_TOKENS,
        HIDDEN,
    )

    return "\n".join(lines).rstrip() + "\n"


def build_decode():
    lines = [
        "MER 32'hC0 64'h01",
        "IER 32'hC8 64'h3F",
        "",
        "# SmolVLM2-500M-Video-Instruct decode",
        "# decode appends current K/V to DRAM cache, then reads prefill+current cache",
        "",
    ]

    emit_layernorm(
        lines,
        "Decode layernorm 1",
        DRAM_DECODE_INPUT,
        HIDDEN,
        DRAM_DECODE_LN1,
        HIDDEN,
        DECODE_TOKENS,
        HIDDEN,
    )

    emit_gemm_matrix(
        lines,
        "Decode Q projection",
        DRAM_DECODE_LN1,
        HIDDEN,
        DRAM_W_Q,
        HIDDEN,
        DRAM_DECODE_Q,
        HIDDEN,
        DECODE_TOKENS,
        HIDDEN,
        HIDDEN,
    )

    emit_gemm_matrix(
        lines,
        "Decode K projection append cache",
        DRAM_DECODE_LN1,
        HIDDEN,
        DRAM_W_K,
        KV_DIM,
        DRAM_DECODE_K_CACHE_T,
        KV_CACHE_STRIDE,
        DECODE_TOKENS,
        KV_DIM,
        HIDDEN,
        transpose_out=True,
        transpose_stride=KV_CACHE_STRIDE,
    )

    emit_gemm_matrix(
        lines,
        "Decode V projection append cache",
        DRAM_DECODE_LN1,
        HIDDEN,
        DRAM_W_V,
        KV_DIM,
        DRAM_DECODE_V_CACHE,
        KV_DIM,
        DECODE_TOKENS,
        KV_DIM,
        HIDDEN,
    )

    emit_attention(lines, "Decode", DRAM_DECODE_Q, DRAM_K_CACHE_T, DRAM_V_CACHE,
                   DECODE_TOKENS, KV_CACHE_STRIDE, KV_CACHE_STRIDE)

    emit_gemm_matrix(
        lines,
        "Decode output projection",
        DRAM_CONTEXT,
        HIDDEN,
        DRAM_W_O,
        HIDDEN,
        DRAM_DECODE_ATTN_OUT,
        HIDDEN,
        DECODE_TOKENS,
        HIDDEN,
        HIDDEN,
    )

    emit_residual_add(
        lines,
        "Decode residual 1",
        DRAM_DECODE_INPUT,
        HIDDEN,
        DRAM_DECODE_ATTN_OUT,
        HIDDEN,
        DRAM_DECODE_RESID1,
        HIDDEN,
        DECODE_TOKENS,
        HIDDEN,
    )

    emit_layernorm(
        lines,
        "Decode layernorm 2",
        DRAM_DECODE_RESID1,
        HIDDEN,
        DRAM_DECODE_LN2,
        HIDDEN,
        DECODE_TOKENS,
        HIDDEN,
    )

    emit_gemm_matrix(
        lines,
        "Decode FFN up projection",
        DRAM_DECODE_LN2,
        HIDDEN,
        DRAM_W_FFN_UP,
        INTERMEDIATE,
        DRAM_DECODE_FFN_UP,
        INTERMEDIATE,
        DECODE_TOKENS,
        INTERMEDIATE,
        HIDDEN,
    )

    emit_activation(
        lines,
        "Decode FFN activation",
        DRAM_DECODE_FFN_UP,
        INTERMEDIATE,
        DRAM_DECODE_FFN_ACT,
        INTERMEDIATE,
        DECODE_TOKENS,
        INTERMEDIATE,
    )

    emit_gemm_matrix(
        lines,
        "Decode FFN down projection",
        DRAM_DECODE_FFN_ACT,
        INTERMEDIATE,
        DRAM_W_FFN_DOWN,
        HIDDEN,
        DRAM_DECODE_FFN_DOWN,
        HIDDEN,
        DECODE_TOKENS,
        HIDDEN,
        INTERMEDIATE,
    )

    emit_residual_add(
        lines,
        "Decode residual 2",
        DRAM_DECODE_RESID1,
        HIDDEN,
        DRAM_DECODE_FFN_DOWN,
        HIDDEN,
        DRAM_DECODE_OUT,
        HIDDEN,
        DECODE_TOKENS,
        HIDDEN,
    )

    return "\n".join(lines).rstrip() + "\n"


def build_config(top_name: str) -> str:
    return f"""# SmolVLM2 benchmark configuration for main/PIMU-Sim
simulator_type = PERFORMANCE
max_sim_cycle = 50000000

core_mode = 0
systolic_array_size = 32
npu_freq = 1.0

mvin_rs1_reg_addr = 32'h00
mvin_rs2_reg_addr = 32'h08
mvout_rs1_reg_addr = 32'h10
mvout_rs2_reg_addr = 32'h18
config_mvin_rs1_reg_addr = 32'h20
config_mvin_rs2_reg_addr = 32'h28
config_mvout_rs1_reg_addr = 32'h30
config_mvout_rs2_reg_addr = 32'h38
config_compute_rs1_reg_addr = 32'h40
config_compute_rs2_reg_addr = 32'h48
config_sfu_rs1_reg_addr = 32'h50
config_sfu_rs2_reg_addr = 32'h58
config_accumulate_rs1_reg_addr = 32'h60
config_accumulate_rs2_reg_addr = 32'h68
compute_sa_rs1_reg_addr = 32'h70
compute_sa_rs2_reg_addr = 32'h78
compute_matadd_rs1_reg_addr = 32'h80
compute_matadd_rs2_reg_addr = 32'h88
compute_matvec_rs1_reg_addr = 32'h90
compute_matvec_rs2_reg_addr = 32'h98
compute_sfu_rs1_reg_addr = 32'ha0
compute_sfu_rs2_reg_addr = 32'ha8
start_reg_addr = 32'hb0
iar_reg_addr = 32'hb8
mer_reg_addr = 32'hc0
ier_reg_addr = 32'hc8
isr_reg_addr = 32'hd0
ipr_reg_addr = 32'hd8

ddr_mode = 1
dramsim_cfg_path = ./external/DRAMsim3/configs/DDR4_8Gb_x8_3200.ini
ddr_freq = 3.2
ddr_bandwidth = 21.3
tCAS = 15
tRCD = 15
tRP = 15
tBURST = 4
BL = 8
ddr_bus_width = 64
ddr_channel_num = 4
ddr_request_queue_depth = 8

sram_bytes = 131072
sram_ports = 2
sram_data_width = 8

acc_bytes = 524288
acc_ports = 2
acc_data_width = 32

enable_power = 1
power_file = ./config/power.txt

inst_file = ./benchmarks/tb_test/tb_{top_name}.txt
enable_log = 0
log_file = ./log/tb_{top_name}_log.txt
"""


def main():
    base = Path(__file__).with_name
    sim_root = Path(__file__).resolve().parents[2]
    cfg_dir = sim_root / "config"
    prefill_path = base("tb_smolvlm2_prefill.txt")
    decode_path = base("tb_smolvlm2_decode.txt")
    prefill_cfg_path = cfg_dir / "smolvlm2_prefill_cfg.txt"
    decode_cfg_path = cfg_dir / "smolvlm2_decode_cfg.txt"
    prefill_path.write_text(build_prefill(), encoding="ascii")
    decode_path.write_text(build_decode(), encoding="ascii")
    prefill_cfg_path.write_text(build_config("smolvlm2_prefill"), encoding="ascii")
    decode_cfg_path.write_text(build_config("smolvlm2_decode"), encoding="ascii")
    print(f"generated {prefill_path}")
    print(f"generated {decode_path}")
    print(f"generated {prefill_cfg_path}")
    print(f"generated {decode_cfg_path}")


if __name__ == "__main__":
    main()
