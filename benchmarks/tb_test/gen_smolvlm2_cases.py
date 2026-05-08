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
ACT_CHUNK = INTERMEDIATE

assert NUM_Q_HEADS % NUM_KV_HEADS == 0
assert HIDDEN == NUM_Q_HEADS * HEAD_DIM
assert KV_DIM == NUM_KV_HEADS * HEAD_DIM

# Prefill writes tokens [0, 31]. Decode appends token 32 and attends over 33
# cached positions. The K cache is transposed, so this is its row stride.
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

BANK_BYTES = 0x00040000
A_BANK0 = 0 * BANK_BYTES
A_BANK1 = 1 * BANK_BYTES
B_BANK0 = 2 * BANK_BYTES
B_BANK1 = 3 * BANK_BYTES
SFU_IN_BANK = 4 * BANK_BYTES
SFU_OUT_BANK = 5 * BANK_BYTES
C_BANK0 = 6 * BANK_BYTES
C_BANK1 = 7 * BANK_BYTES
OUTPUT_BANKS = [C_BANK0, C_BANK1]
BARRIER_BANKS = [SFU_IN_BANK, SFU_OUT_BANK]
BARRIER_REPEAT = 2

ACC_BANK0 = 0x00000000
ACC_BANK1 = 0x00080000
DRAM_BARRIER = 0x1F000000


def inst_hex(high: int, low: int) -> str:
    return f"{high:016x}{low:016x}"


def emit_dma(opcode: int, data_type: int, precision: int, is_quant: int, source: int, is_bias: int,
             col_num: int, row_num: int, sram_addr: int, dram_addr: int,
             sram_stride: int, dram_stride: int, qscale: int = 0, qshift: int = 0, qzp: int = 0):
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
    low1 = sram_stride | (dram_stride << 16) | (qscale << 32) | (qshift << 48)
    high1 = qzp
    return [inst_hex(high0, low0), inst_hex(high1, low1)]


def emit_mvin(dram_addr: int, local_addr: int, row_num: int, col_num: int,
              dram_stride: int, sram_stride: int, to_acc: bool = False):
    return emit_dma(
        opcode=0x01,
        data_type=2 if to_acc else 0,
        precision=1,
        is_quant=0,
        source=1 if to_acc else 0,
        is_bias=0,
        col_num=col_num - 1,
        row_num=row_num - 1,
        sram_addr=local_addr,
        dram_addr=dram_addr,
        sram_stride=sram_stride,
        dram_stride=dram_stride,
    )


def emit_mvout(local_addr: int, dram_addr: int, row_num: int, col_num: int,
               dram_stride: int, sram_stride: int, from_acc: bool = False):
    return emit_dma(
        opcode=0x02,
        data_type=2 if from_acc else 0,
        precision=1,
        is_quant=0,
        source=1 if from_acc else 0,
        is_bias=0,
        col_num=col_num - 1,
        row_num=row_num - 1,
        sram_addr=local_addr,
        dram_addr=dram_addr,
        sram_stride=sram_stride,
        dram_stride=dram_stride,
    )


def emit_sa(config2: int, bias_stride: int, output_stride: int, biaspsum_addr: int, output_addr: int,
            a_addr: int, a_col_num: int, a_row_num: int, a_stride: int,
            b_addr: int, b_col_num: int, b_row_num: int, b_stride: int):
    low0 = 0x03
    high0 = 1
    low1 = (config2 & 0x1FFF) | (bias_stride << 32) | (output_stride << 48)
    high1 = biaspsum_addr | (output_addr << 32)
    low2 = a_addr | ((a_col_num - 1) << 32) | ((a_row_num - 1) << 43) | (a_stride << 48)
    high2 = b_addr | ((b_col_num - 1) << 32) | ((b_row_num - 1) << 37) | (b_stride << 48)
    return [inst_hex(high0, low0), inst_hex(high1, low1), inst_hex(high2, low2)]


def emit_matadd(a_addr: int, b_addr: int, out_addr: int, row_num: int, col_num: int):
    low0 = 0x04 | ((col_num - 1) << 8) | ((row_num - 1) << 16) | (a_addr << 32)
    high0 = b_addr | (out_addr << 32)
    return [inst_hex(high0, low0), inst_hex(0, 0), inst_hex(0, 0)]


def emit_sfu(optype: int, input_addr: int, output_addr: int, row_num: int, col_num: int, is_quant: int = 0):
    low0 = 0x05 | (optype << 8) | (is_quant << 16) | ((col_num - 1) << 32) | ((row_num - 1) << 48)
    high0 = input_addr | (output_addr << 32)
    return [inst_hex(high0, low0), inst_hex(0, 0)]


def tile_ranges(size: int):
    return range(0, size, TILE)


def tile_size(size: int, offset: int) -> int:
    return min(TILE, size - offset)


def dram_addr(base: int, row_offset: int, col_offset: int, stride: int) -> int:
    return base + row_offset * stride + col_offset


def output_bank(row_offset: int, col_offset: int, cols: int, bank_tag: int = 0, bank_pool=None) -> int:
    bank_pool = OUTPUT_BANKS if bank_pool is None else bank_pool
    nt = (cols + TILE - 1) // TILE
    tile_id = (row_offset // TILE) * nt + (col_offset // TILE)
    return bank_pool[(bank_tag + tile_id) % len(bank_pool)]


def other_output_bank(bank: int) -> int:
    return C_BANK1 if bank == C_BANK0 else C_BANK0


def emit_gemm_matrix(lines, name: str, a_base: int, a_stride: int, b_base: int, b_stride: int,
                     out_base: int, out_stride: int, rows: int, cols: int, depth: int,
                     transpose_out: bool = False, transpose_stride: int = 0, bank_tag: int = 0,
                     bank_pool=None):
    lines.append(f"# {name}")
    for i in tile_ranges(rows):
        row_t = tile_size(rows, i)
        for j in tile_ranges(cols):
            col_t = tile_size(cols, j)
            c_bank = output_bank(i, j, cols, bank_tag, bank_pool)
            t_bank = SFU_OUT_BANK
            lines.append(f"# GEMM tile ({i // TILE}, {j // TILE})")

            for k in tile_ranges(depth):
                k_t = tile_size(depth, k)
                ping = (k // TILE) & 0x1
                a_bank = A_BANK0 if ping == 0 else A_BANK1
                b_bank = B_BANK0 if ping == 0 else B_BANK1

                lines.extend(
                    emit_mvin(
                        dram_addr(a_base, i, k, a_stride),
                        a_bank,
                        row_t,
                        k_t,
                        a_stride,
                        k_t,
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
                    )
                )

                if depth <= TILE:
                    cfg2 = 0
                    biaspsum_addr = 0
                    output_addr = c_bank
                elif k == 0:
                    cfg2 = 1 << 3
                    biaspsum_addr = 0
                    output_addr = ACC_BANK1
                elif k + TILE < depth:
                    cfg2 = (1 << 3) | (1 << 6)
                    biaspsum_addr = ACC_BANK1
                    output_addr = ACC_BANK1
                else:
                    cfg2 = 1 << 6
                    biaspsum_addr = ACC_BANK1
                    output_addr = c_bank

                lines.append(f"# SA tile ({i // TILE}, {j // TILE}, {k // TILE})")
                lines.extend(
                    emit_sa(
                        config2=cfg2,
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
                lines.extend(emit_sfu(8, c_bank, t_bank, row_t, col_t))
                lines.extend(
                    emit_mvout(
                        t_bank,
                        dram_addr(out_base, j, i, transpose_stride),
                        col_t,
                        row_t,
                        transpose_stride,
                        row_t,
                    )
                )
            else:
                lines.append(f"# MVOUT tile ({i // TILE}, {j // TILE})")
                lines.extend(
                    emit_mvout(
                        c_bank,
                        dram_addr(out_base, i, j, out_stride),
                        row_t,
                        col_t,
                        out_stride,
                        col_t,
                    )
                )


def emit_sfu_matrix(lines, name: str, optype: int, src_base: int, src_stride: int,
                    dst_base: int, dst_stride: int, rows: int, cols: int, chunk_cols: int,
                    bank_tag: int = 0):
    lines.append(f"# {name}")
    for col_off in range(0, cols, chunk_cols):
        chunk = min(chunk_cols, cols - col_off)
        if cols > chunk_cols:
            lines.append(f"# SFU chunk {col_off // chunk_cols}")
        lines.extend(
            emit_mvin(
                src_base + col_off,
                SFU_IN_BANK,
                rows,
                chunk,
                src_stride,
                chunk,
            )
        )
        lines.extend(emit_sfu(optype, SFU_IN_BANK, SFU_OUT_BANK, rows, chunk))
        lines.extend(
            emit_mvout(
                SFU_OUT_BANK,
                dst_base + col_off,
                rows,
                chunk,
                dst_stride,
                chunk,
            )
        )


def emit_layernorm(lines, name: str, src_base: int, src_stride: int, dst_base: int, dst_stride: int,
                   rows: int, cols: int, bank_tag: int = 0):
    emit_sfu_matrix(lines, name, 2, src_base, src_stride, dst_base, dst_stride, rows, cols, cols, bank_tag)


def emit_softmax(lines, name: str, src_base: int, src_stride: int, dst_base: int, dst_stride: int,
                 rows: int, cols: int, bank_tag: int = 0):
    emit_sfu_matrix(lines, name, 0, src_base, src_stride, dst_base, dst_stride, rows, cols, cols, bank_tag)


def emit_activation(lines, name: str, src_base: int, src_stride: int, dst_base: int, dst_stride: int,
                    rows: int, cols: int, bank_tag: int = 0):
    emit_sfu_matrix(lines, name, 1, src_base, src_stride, dst_base, dst_stride, rows, cols, ACT_CHUNK, bank_tag)


def emit_residual_add(lines, name: str, a_base: int, a_stride: int, b_base: int, b_stride: int,
                      out_base: int, out_stride: int, rows: int, cols: int, bank_tag: int = 0):
    lines.append(f"# {name}")
    for i in tile_ranges(rows):
        row_t = tile_size(rows, i)
        for j in tile_ranges(cols):
            col_t = tile_size(cols, j)
            out_bank = output_bank(i, j, cols, bank_tag)
            lines.append(f"# MATADD tile ({i // TILE}, {j // TILE})")
            lines.extend(
                emit_mvin(
                    dram_addr(a_base, i, j, a_stride),
                    ACC_BANK0,
                    row_t,
                    col_t,
                    a_stride,
                    col_t,
                    to_acc=True,
                )
            )
            lines.extend(
                emit_mvin(
                    dram_addr(b_base, i, j, b_stride),
                    ACC_BANK1,
                    row_t,
                    col_t,
                    b_stride,
                    col_t,
                    to_acc=True,
                )
            )
            lines.extend(emit_matadd(ACC_BANK0, ACC_BANK1, out_bank, row_t, col_t))
            lines.extend(
                emit_mvout(
                    out_bank,
                    dram_addr(out_base, i, j, out_stride),
                    row_t,
                    col_t,
                    out_stride,
                    col_t,
                )
            )


def emit_acc_barrier(lines, name: str, repeats: int = BARRIER_REPEAT):
    lines.append(f"# {name}")
    for idx in range(repeats):
        out_bank = BARRIER_BANKS[idx % len(BARRIER_BANKS)]
        lines.append(f"# ACC/MATADD barrier {idx}")
        lines.extend(emit_mvin(DRAM_LN1, ACC_BANK0, TILE, TILE, HIDDEN, TILE, to_acc=True))
        lines.extend(emit_mvin(DRAM_W_Q, ACC_BANK1, TILE, TILE, HIDDEN, TILE, to_acc=True))
        lines.extend(emit_matadd(ACC_BANK0, ACC_BANK1, out_bank, TILE, TILE))
        lines.extend(
            emit_mvout(
                out_bank,
                DRAM_BARRIER + idx * TILE * TILE,
                TILE,
                TILE,
                TILE,
                TILE,
            )
        )


def emit_attention(lines, prefix: str, q_base_root: int, k_cache_root: int, v_cache_root: int,
                   rows: int, cache_cols: int, cache_stride: int, bank_tag: int = 0):
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
            bank_tag=bank_tag,
            bank_pool=[C_BANK0],
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
            bank_tag=bank_tag,
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
            bank_tag=bank_tag,
            bank_pool=[C_BANK1],
        )


def build_prefill():
    lines = [
        "# SmolVLM2-500M-Video-Instruct prefill transformer block",
        "# fixed new_spm map: bank0/1=A, bank2/3=B, bank4/5=SFU, bank6/7=SA/MATADD output",
        "# fixed DMA map: DMA1=A, DMA2=B, DMA3=ACC, DMA4=bank4-7 MVIN, DMA5/6=bank4-7 MVOUT",
        "# K cache is written transposed with one reserved decode slot; V cache is sequence-major.",
        "",
    ]

    emit_layernorm(lines, "Prefill layernorm 1", DRAM_PREFILL_INPUT, HIDDEN, DRAM_LN1, HIDDEN, PREFILL_TOKENS, HIDDEN, bank_tag=0)
    emit_gemm_matrix(lines, "Prefill Q projection", DRAM_LN1, HIDDEN, DRAM_W_Q, HIDDEN, DRAM_PREFILL_Q, HIDDEN, PREFILL_TOKENS, HIDDEN, HIDDEN, bank_tag=3)
    emit_gemm_matrix(lines, "Prefill K projection", DRAM_LN1, HIDDEN, DRAM_W_K, KV_DIM, DRAM_K_CACHE_T, KV_CACHE_STRIDE, PREFILL_TOKENS, KV_DIM, HIDDEN, transpose_out=True, transpose_stride=KV_CACHE_STRIDE, bank_tag=6)
    emit_gemm_matrix(lines, "Prefill V projection", DRAM_LN1, HIDDEN, DRAM_W_V, KV_DIM, DRAM_V_CACHE, KV_DIM, PREFILL_TOKENS, KV_DIM, HIDDEN, bank_tag=9)

    emit_attention(lines, "Prefill", DRAM_PREFILL_Q, DRAM_K_CACHE_T, DRAM_V_CACHE, PREFILL_TOKENS, PREFILL_TOKENS, KV_CACHE_STRIDE, bank_tag=12)

    emit_gemm_matrix(lines, "Prefill output projection", DRAM_CONTEXT, HIDDEN, DRAM_W_O, HIDDEN, DRAM_ATTN_OUT, HIDDEN, PREFILL_TOKENS, HIDDEN, HIDDEN, bank_tag=60)
    emit_residual_add(lines, "Prefill residual 1", DRAM_PREFILL_INPUT, HIDDEN, DRAM_ATTN_OUT, HIDDEN, DRAM_RESID1, HIDDEN, PREFILL_TOKENS, HIDDEN, bank_tag=63)
    emit_layernorm(lines, "Prefill layernorm 2", DRAM_RESID1, HIDDEN, DRAM_LN2, HIDDEN, PREFILL_TOKENS, HIDDEN, bank_tag=66)
    emit_gemm_matrix(lines, "Prefill FFN up projection", DRAM_LN2, HIDDEN, DRAM_W_FFN_UP, INTERMEDIATE, DRAM_FFN_UP, INTERMEDIATE, PREFILL_TOKENS, INTERMEDIATE, HIDDEN, bank_tag=69)
    emit_activation(lines, "Prefill FFN activation", DRAM_FFN_UP, INTERMEDIATE, DRAM_FFN_ACT, INTERMEDIATE, PREFILL_TOKENS, INTERMEDIATE, bank_tag=72)
    emit_gemm_matrix(lines, "Prefill FFN down projection", DRAM_FFN_ACT, INTERMEDIATE, DRAM_W_FFN_DOWN, HIDDEN, DRAM_FFN_DOWN, HIDDEN, PREFILL_TOKENS, HIDDEN, INTERMEDIATE, bank_tag=75)
    emit_residual_add(lines, "Prefill residual 2", DRAM_RESID1, HIDDEN, DRAM_FFN_DOWN, HIDDEN, DRAM_PREFILL_OUT, HIDDEN, PREFILL_TOKENS, HIDDEN, bank_tag=78)
    return "\n".join(lines).rstrip() + "\n"


def build_decode():
    lines = [
        "# SmolVLM2-500M-Video-Instruct decode transformer block",
        "# Decode appends current K/V to the DRAM KV cache and attends over prefill+current tokens.",
        "# All DMA routes are chosen by fixed bank addresses, so bank placement is part of the test.",
        "",
    ]

    emit_layernorm(lines, "Decode layernorm 1", DRAM_DECODE_INPUT, HIDDEN, DRAM_DECODE_LN1, HIDDEN, DECODE_TOKENS, HIDDEN, bank_tag=0)
    emit_gemm_matrix(lines, "Decode Q projection", DRAM_DECODE_LN1, HIDDEN, DRAM_W_Q, HIDDEN, DRAM_DECODE_Q, HIDDEN, DECODE_TOKENS, HIDDEN, HIDDEN, bank_tag=3)
    emit_gemm_matrix(lines, "Decode K projection append cache", DRAM_DECODE_LN1, HIDDEN, DRAM_W_K, KV_DIM, DRAM_DECODE_K_CACHE_T, KV_CACHE_STRIDE, DECODE_TOKENS, KV_DIM, HIDDEN, transpose_out=True, transpose_stride=KV_CACHE_STRIDE, bank_tag=6)
    emit_gemm_matrix(lines, "Decode V projection append cache", DRAM_DECODE_LN1, HIDDEN, DRAM_W_V, KV_DIM, DRAM_DECODE_V_CACHE, KV_DIM, DECODE_TOKENS, KV_DIM, HIDDEN, bank_tag=9)

    emit_attention(lines, "Decode", DRAM_DECODE_Q, DRAM_K_CACHE_T, DRAM_V_CACHE, DECODE_TOKENS, KV_CACHE_STRIDE, KV_CACHE_STRIDE, bank_tag=12)

    emit_gemm_matrix(lines, "Decode output projection", DRAM_CONTEXT, HIDDEN, DRAM_W_O, HIDDEN, DRAM_DECODE_ATTN_OUT, HIDDEN, DECODE_TOKENS, HIDDEN, HIDDEN, bank_tag=60)
    emit_residual_add(lines, "Decode residual 1", DRAM_DECODE_INPUT, HIDDEN, DRAM_DECODE_ATTN_OUT, HIDDEN, DRAM_DECODE_RESID1, HIDDEN, DECODE_TOKENS, HIDDEN, bank_tag=63)
    emit_layernorm(lines, "Decode layernorm 2", DRAM_DECODE_RESID1, HIDDEN, DRAM_DECODE_LN2, HIDDEN, DECODE_TOKENS, HIDDEN, bank_tag=66)
    emit_gemm_matrix(lines, "Decode FFN up projection", DRAM_DECODE_LN2, HIDDEN, DRAM_W_FFN_UP, INTERMEDIATE, DRAM_DECODE_FFN_UP, INTERMEDIATE, DECODE_TOKENS, INTERMEDIATE, HIDDEN, bank_tag=69)
    emit_activation(lines, "Decode FFN activation", DRAM_DECODE_FFN_UP, INTERMEDIATE, DRAM_DECODE_FFN_ACT, INTERMEDIATE, DECODE_TOKENS, INTERMEDIATE, bank_tag=72)
    emit_gemm_matrix(lines, "Decode FFN down projection", DRAM_DECODE_FFN_ACT, INTERMEDIATE, DRAM_W_FFN_DOWN, HIDDEN, DRAM_DECODE_FFN_DOWN, HIDDEN, DECODE_TOKENS, HIDDEN, INTERMEDIATE, bank_tag=75)
    emit_residual_add(lines, "Decode residual 2", DRAM_DECODE_RESID1, HIDDEN, DRAM_DECODE_FFN_DOWN, HIDDEN, DRAM_DECODE_OUT, HIDDEN, DECODE_TOKENS, HIDDEN, bank_tag=78)
    return "\n".join(lines).rstrip() + "\n"


def build_config(top_name: str) -> str:
    return f"""# SmolVLM2 benchmark configuration for new_spm/PIMU-Sim
simulator_type = PERFORMANCE
max_sim_cycle = 30000000

core_mode = 0
systolic_array_size = 32
npu_freq = 1.0

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

sram_bank_num = 8
sram_bank_bytes = 262144
sram_ports = 2
sram_data_width = 8

acc_bank_num = 2
bias_bank_num = 1
acc_bytes = 1048576
acc_ports = 2
acc_data_width = 32

inst_cache_size = 1024
inst_queue_depth = 2

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
