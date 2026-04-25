#!/usr/bin/env python3
"""
Generate a serial resnet_block benchmark for the main PIMU-Sim branch.

Target flow per output tile:
  1. MVIN A(i, k) -> SPM
  2. MVIN B(k, j) -> SPM
  3. SA
     - first k tile: write psum to ACC
     - middle k tile: accumulate in ACC
     - last k tile: accumulate from ACC and write final C tile to SPM
  4. MVOUT C(i, j) -> DRAM temp
  5. MVIN residual A(i, j) -> ACC
  6. MVIN temp C(i, j) -> ACC
  7. MATADD -> SPM D tile
  8. MVOUT D(i, j) -> DRAM output

This keeps the same tiled resnet_block structure as the other branches,
but maps it onto the main branch's serial register-write benchmark format.
"""

from pathlib import Path

TILE = 32

M = 256
N = 256
K = 256

assert M % TILE == 0 and N % TILE == 0 and K % TILE == 0

MT = M // TILE
NT = N // TILE
KT = K // TILE

LDA = K
LDB = N
LDC = N

DRAM_A_BASE = 0x01000000
DRAM_B_BASE = 0x02000000
DRAM_TMP_BASE = 0x03000000
DRAM_D_BASE = 0x04000000

# Flat SPM layout in the main branch. The architecture is serial, so we reuse
# one tile buffer per role instead of banked ping-pong buffers.
A_BUF = 0x00000000
B_BUF = 0x00001000
C_BUF = 0x00002000
D_BUF = 0x00003000

# ACC SRAM stores 32-bit psums / MATADD inputs.
ACC_A_BUF = 0x00000000
ACC_PSUM_BUF = 0x00001000

TMP_TILE_BYTES = TILE * TILE * 4

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
    "CFG_ACC_RS1": 0x60,
    "CFG_ACC_RS2": 0x68,
    "COMPUTE_SA_RS1": 0x70,
    "COMPUTE_SA_RS2": 0x78,
    "COMPUTE_MATADD_RS1": 0x80,
    "COMPUTE_MATADD_RS2": 0x88,
    "START": 0xB0,
    "MER": 0xC0,
    "IER": 0xC8,
}

START_BITS = {
    "MVIN": 0x1,
    "MVOUT": 0x2,
    "SA": 0x4,
    "MATADD": 0x8,
}


def word64(value: int) -> str:
    hi = (value >> 32) & 0xFFFFFFFF
    lo = value & 0xFFFFFFFF
    return f"64'h{hi:08x}_{lo:08x}"


def reg32(value: int) -> str:
    return f"32'h{value:02x}"


def emit_write(op_name: str, reg_addr: int, value: int) -> str:
    return f"{op_name} {reg32(reg_addr)} {word64(value)}"


def emit_start(bit_name: str):
    return [
        emit_write("ENR", ADDR["START"], START_BITS[bit_name]),
        f"# IAR {reg32(0xB8)} {word64(START_BITS[bit_name])}",
    ]


def dram_addr_a(i_tile: int, k_tile: int) -> int:
    return DRAM_A_BASE + (i_tile * TILE) * LDA + (k_tile * TILE)


def dram_addr_b(k_tile: int, j_tile: int) -> int:
    return DRAM_B_BASE + (k_tile * TILE) * LDB + (j_tile * TILE)


def dram_addr_residual_a(i_tile: int, j_tile: int) -> int:
    return DRAM_A_BASE + (i_tile * TILE) * LDA + (j_tile * TILE)


def dram_addr_tmp(i_tile: int, j_tile: int) -> int:
    return DRAM_TMP_BASE + (i_tile * NT + j_tile) * TMP_TILE_BYTES


def dram_addr_d(i_tile: int, j_tile: int) -> int:
    return DRAM_D_BASE + (i_tile * TILE) * LDC + (j_tile * TILE)


def encode_mvin_cfg(dram_stride: int, sram_stride: int, to_acc: bool) -> int:
    flags = 0
    if to_acc:
        flags |= 1 << 1   # int32 local path
        flags |= 1 << 5   # destination = ACC
    flags |= 1 << 2       # precision selector used by the main branch testbenches
    return (dram_stride << 48) | (sram_stride << 32) | flags


def encode_mvout_cfg(dram_stride: int, sram_stride: int, from_acc: bool) -> int:
    flags = 0
    if from_acc:
        flags |= 1 << 1   # int32 local path
        flags |= 1 << 5   # source = ACC
    flags |= 1 << 2
    return (dram_stride << 48) | (sram_stride << 32) | flags


def encode_sa_compute_rs1(base_addr: int, col_num: int, row_num: int, stride: int) -> int:
    return (
        base_addr
        | ((col_num - 1) << 32)
        | ((row_num - 1) << 43)
        | (stride << 48)
    )


def encode_sa_compute_rs2(base_addr: int, col_num: int, row_num: int, stride: int) -> int:
    return (
        base_addr
        | ((col_num - 1) << 32)
        | ((row_num - 1) << 37)
        | (stride << 48)
    )


def emit_mvin(dram_addr: int, local_addr: int, dram_stride: int, sram_stride: int, to_acc: bool):
    return [
        emit_write("mvin", ADDR["MVIN_RS1"], ((TILE - 1) << 32) | dram_addr),
        emit_write("mvin", ADDR["MVIN_RS2"], ((TILE - 1) << 32) | local_addr),
        emit_write("config_mvin", ADDR["CFG_MVIN_RS1"], encode_mvin_cfg(dram_stride, sram_stride, to_acc)),
        emit_write("config_mvin", ADDR["CFG_MVIN_RS2"], 0x0000000100000000),
        *emit_start("MVIN"),
    ]


def emit_mvout(local_addr: int, dram_addr: int, dram_stride: int, sram_stride: int, from_acc: bool):
    return [
        emit_write("mvout", ADDR["MVOUT_RS1"], ((TILE - 1) << 32) | local_addr),
        emit_write("mvout", ADDR["MVOUT_RS2"], ((TILE - 1) << 32) | dram_addr),
        emit_write("config_mvout", ADDR["CFG_MVOUT_RS1"], encode_mvout_cfg(dram_stride, sram_stride, from_acc)),
        emit_write("config_mvout", ADDR["CFG_MVOUT_RS2"], 0x0000000100000000),
        *emit_start("MVOUT"),
    ]


def emit_sa(is_acc: bool, acc_dest: bool, biaspsum_addr: int, output_addr: int):
    cfg_compute_rs1 = 0x00000001 | ((1 if acc_dest else 0) << 27)
    cfg_acc_rs1 = (TILE << 32) | biaspsum_addr

    acc_hi = TILE | ((1 if is_acc else 0) << 16)
    cfg_acc_rs2 = (acc_hi << 32) | output_addr

    return [
        emit_write("config_compute", ADDR["CFG_COMPUTE_RS1"], cfg_compute_rs1),
        emit_write("config_compute", ADDR["CFG_COMPUTE_RS2"], 0x0000000100000000),
        emit_write("config_accumulate", ADDR["CFG_ACC_RS1"], cfg_acc_rs1),
        emit_write("config_accumulate", ADDR["CFG_ACC_RS2"], cfg_acc_rs2),
        emit_write("compute_sa", ADDR["COMPUTE_SA_RS1"], encode_sa_compute_rs1(A_BUF, TILE, TILE, TILE)),
        emit_write("compute_sa", ADDR["COMPUTE_SA_RS2"], encode_sa_compute_rs2(B_BUF, TILE, TILE, TILE)),
        *emit_start("SA"),
    ]


def emit_matadd():
    rs1 = (ACC_PSUM_BUF << 32) | ACC_A_BUF
    rs2 = ((TILE - 1) << 40) | ((TILE - 1) << 32) | D_BUF
    return [
        emit_write("config_compute", ADDR["CFG_COMPUTE_RS1"], 0x0000000000000001),
        emit_write("config_compute", ADDR["CFG_COMPUTE_RS2"], 0x0000000100000000),
        emit_write("compute_matadd", ADDR["COMPUTE_MATADD_RS1"], rs1),
        emit_write("compute_matadd", ADDR["COMPUTE_MATADD_RS2"], rs2),
        *emit_start("MATADD"),
    ]


def generate():
    lines = [
        "MER 32'hc0 64'h00000000_00000001",
        "IER 32'hc8 64'h00000000_0000003f",
        "",
        "# Serial resnet_block benchmark for main/PIMU-Sim",
        "# D = A * B + A",
        "# A/B GEMM is tiled in 32x32x32, then residual add is done by MATADD.",
        "# The main branch is serial, so each output tile reuses the same local buffers.",
        "",
    ]

    for i in range(MT):
        for j in range(NT):
            lines.append(f"# ---- Output Tile ({i}, {j}) ----")

            for k in range(KT):
                lines.append(f"# MVIN A tile ({i}, {k}) -> SPM")
                lines.extend(
                    emit_mvin(
                        dram_addr=dram_addr_a(i, k),
                        local_addr=A_BUF,
                        dram_stride=LDA,
                        sram_stride=TILE,
                        to_acc=False,
                    )
                )
                lines.append("")

                lines.append(f"# MVIN B tile ({k}, {j}) -> SPM")
                lines.extend(
                    emit_mvin(
                        dram_addr=dram_addr_b(k, j),
                        local_addr=B_BUF,
                        dram_stride=LDB,
                        sram_stride=TILE,
                        to_acc=False,
                    )
                )
                lines.append("")

                if k == 0:
                    lines.append(f"# SA tile ({i}, {j}, {k}) -> ACC psum")
                    lines.extend(
                        emit_sa(
                            is_acc=False,
                            acc_dest=True,
                            biaspsum_addr=0x00000000,
                            output_addr=ACC_PSUM_BUF,
                        )
                    )
                elif k == KT - 1:
                    lines.append(f"# SA tile ({i}, {j}, {k}) -> final C tile in SPM")
                    lines.extend(
                        emit_sa(
                            is_acc=True,
                            acc_dest=False,
                            biaspsum_addr=ACC_PSUM_BUF,
                            output_addr=C_BUF,
                        )
                    )
                else:
                    lines.append(f"# SA tile ({i}, {j}, {k}) -> accumulate in ACC")
                    lines.extend(
                        emit_sa(
                            is_acc=True,
                            acc_dest=True,
                            biaspsum_addr=ACC_PSUM_BUF,
                            output_addr=ACC_PSUM_BUF,
                        )
                    )
                lines.append("")

            lines.append(f"# MVOUT C tile ({i}, {j}) -> DRAM temp")
            lines.extend(
                emit_mvout(
                    local_addr=C_BUF,
                    dram_addr=dram_addr_tmp(i, j),
                    dram_stride=TILE,
                    sram_stride=TILE,
                    from_acc=False,
                )
            )
            lines.append("")

            lines.append(f"# MVIN residual A tile ({i}, {j}) -> ACC")
            lines.extend(
                emit_mvin(
                    dram_addr=dram_addr_residual_a(i, j),
                    local_addr=ACC_A_BUF,
                    dram_stride=LDA,
                    sram_stride=TILE,
                    to_acc=True,
                )
            )
            lines.append("")

            lines.append(f"# MVIN temp C tile ({i}, {j}) -> ACC")
            lines.extend(
                emit_mvin(
                    dram_addr=dram_addr_tmp(i, j),
                    local_addr=ACC_PSUM_BUF,
                    dram_stride=TILE,
                    sram_stride=TILE,
                    to_acc=True,
                )
            )
            lines.append("")

            lines.append(f"# MATADD tile ({i}, {j}) -> SPM D")
            lines.extend(emit_matadd())
            lines.append("")

            lines.append(f"# MVOUT D tile ({i}, {j}) -> DRAM output")
            lines.extend(
                emit_mvout(
                    local_addr=D_BUF,
                    dram_addr=dram_addr_d(i, j),
                    dram_stride=LDC,
                    sram_stride=TILE,
                    from_acc=False,
                )
            )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main():
    out_file = Path(__file__).with_name("tb_resnet_block.txt")
    out_file.write_text(generate(), encoding="ascii")
    print(f"generated {out_file}")


if __name__ == "__main__":
    main()
