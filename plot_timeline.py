from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import plotly.io as pio


# ==========================================================
# User Config
# 只需要改这里，然后直接运行：python3 plot_timeline.py
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "log": BASE_DIR / "log/tb_resnet_block_log.txt",
    "out_html": BASE_DIR / "log/tb_resnet_block_timeline.html",
    "out_png": None,  # 例如: BASE_DIR / "log/tb_resnet_block_timeline.png"
    "title": "PIMU-Sim Execution Timeline",
    "start_cycle": 0,
    "end_cycle": 2000,           # None 表示一直到日志结束
    "lanes": ["DMA1", "DMA2", "DMA3", "SA", "ACC", "SFU", "DDR"],
    "show_markers": True,
    "show_predictions": True,
    "show_waits": True,
    "show_dependencies": True,
    "show_dependency_labels": True,
    "only_active_lanes": True,
    "html_include_plotlyjs": "cdn",  # "cdn" / True
    "max_dependency_per_target": 3,  # GEMM 累加时可能同时依赖 A / B / psum
}


# ==========================================================
# Data Model
# ==========================================================
@dataclass
class Interval:
    lane: str
    start: int
    end: int
    label: str
    color: str
    kind: str = "exec"   # exec / wait
    note: str = ""
    reads: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()


@dataclass
class Marker:
    lane: str
    cycle: int
    label: str
    kind: str            # start / finish / issue / retire
    color: str


@dataclass
class Prediction:
    lane: str
    start: int
    predicted_end: int
    label: str
    color: str


@dataclass
class Dependency:
    src_lane: str
    src_cycle: int
    src_label: str
    dst_lane: str
    dst_cycle: int
    dst_label: str
    label: str
    color: str


COLOR_MAP = {
    "MVIN": "#6F96C4",
    "MVOUT": "#E9A062",
    "GEMM": "#7CB874",
    "CONV": "#D77C71",
    "Matadd": "#C08EBB",
    "Softmax": "#7CBDB8",
    "Layernorm": "#AD9078",
    "Gelu": "#E9A1B0",
    "Transpose": "#B9C2D0",
    "Pool_max": "#D8C46A",
    "Resample": "#9C8EE4",
    "LD_REQ": "#87A9D1",
    "ST_REQ": "#EDB07B",
    "EventSyncWait": "#D8D3CB",
    "DDR": "#A9B4C2",
    "PRED": "#6B7280",
    "DEP": "#C62828",
}


def pick_color(label: str, default: str = "#999999") -> str:
    return COLOR_MAP.get(label, default)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(153,153,153,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def lane_band_color(index: int) -> str:
    return "rgba(116, 136, 160, 0.035)" if index % 2 == 0 else "rgba(116, 136, 160, 0.015)"


def make_resource(space: str, addr: int) -> str:
    return f"{space}:0x{addr:08x}"


def ordered_unique_resources(resources: Tuple[str, ...] | List[str]) -> Tuple[str, ...]:
    seen = set()
    ordered = []
    for resource in resources:
        if resource in seen:
            continue
        seen.add(resource)
        ordered.append(resource)
    return tuple(ordered)


def quadratic_bezier_point(
    start: Tuple[float, float],
    ctrl: Tuple[float, float],
    end: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    one_minus_t = 1.0 - t
    x = one_minus_t * one_minus_t * start[0] + 2 * one_minus_t * t * ctrl[0] + t * t * end[0]
    y = one_minus_t * one_minus_t * start[1] + 2 * one_minus_t * t * ctrl[1] + t * t * end[1]
    return (x, y)


def dependency_curve_geometry(
    dep: Dependency,
    y_pos: Dict[str, int],
    route_index: int,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    start = (float(dep.src_cycle), float(y_pos[dep.src_lane]))
    end = (float(dep.dst_cycle), float(y_pos[dep.dst_lane]))
    lane_gap = abs(start[1] - end[1])
    curve_lift = 0.92 + 0.30 * lane_gap + 0.22 * (route_index % 3)
    ctrl = ((start[0] + end[0]) / 2.0, max(start[1], end[1]) + curve_lift)
    label_pt = quadratic_bezier_point(start, ctrl, end, 0.52)
    arrow_tail = quadratic_bezier_point(start, ctrl, end, 0.90)
    return start, ctrl, end, label_pt, arrow_tail


# ==========================================================
# Parser
# ==========================================================
class TimelineParser:
    cycle_re = re.compile(r"\[Core Cycle:\s*(\d+)\]")

    dma_op_re = re.compile(r"\[DMA\]\s+(MVIN|MVOUT):")
    dma_mvin_re = re.compile(
        r"\[DMA\]\s+MVIN:\s+send\s+LD_REQ,\s+src=0x([0-9a-fA-F]+)\s+dst=0x([0-9a-fA-F]+).*dest=(SPM|ACC)"
    )
    dma_mvout_re = re.compile(
        r"\[DMA\]\s+MVOUT:\s+send\s+ST_REQ,\s+src=0x([0-9a-fA-F]+)\s+dst=0x([0-9a-fA-F]+).*source=(SPM|ACC)"
    )
    dma_issue_re = re.compile(r"\[Core\]\s+DMA\s+(\d)\s+Queue Issued")
    dma_retire_re = re.compile(r"\[Core\]\s+DMA\s+(\d)\s+Instruction Retired")
    sa_queue_word_re = re.compile(
        r"\[SAInstQueue\]\s+In Queue\s*:\s+word=0x([0-9a-fA-F]{16})([0-9a-fA-F]{16}),\s+inst_count=(\d+)/(\d+)"
    )

    sa_block_detail_re = re.compile(
        r"\[SAInstQueue\]\s+Inst Issue Blocked by Event Sync Unit: .*?opcode=0x([0-9a-fA-F]+)"
        r".*?is_acc=(\d).*?is_bias=(\d).*?acc_dest=(\d).*?a_keep=(\d).*?b_keep=(\d).*?bias_keep=(\d)"
        r".*?a_addr=0x([0-9a-fA-F]+)\s+b_addr=0x([0-9a-fA-F]+)"
        r"\s+biaspsum_addr=0x([0-9a-fA-F]+)\s+output_addr=0x([0-9a-fA-F]+)"
    )
    sa_start_re = re.compile(r"\[SystolicArray\]\s+(GEMM|CONV)\s+Start.*remaining\s*=\s*(\d+)")
    sa_issue_re = re.compile(r"\[Core\]\s+SA Queue Issued")
    sa_finish_re = re.compile(r"\[Core\]\s+SA Finished")
    sa_retire_re = re.compile(r"\[Core\]\s+SA Instruction Retired")

    acc_start_re = re.compile(r"\[Accumulator\]\s+(Matadd|GEMM|CONV)\s+Start.*remaining\s*=\s*(\d+)")
    sfu_start_re = re.compile(r"\[SFU\]\s+([A-Za-z_]+)\s+Start.*remaining\s*=\s*(\d+)")
    sfu_issue_re = re.compile(r"\[Core\]\s+SFU Queue Issued")
    sfu_retire_re = re.compile(r"\[Core\]\s+SFU Instruction Retired")

    ddr_start_re = re.compile(r"\[DDR\]\s+start request type=(\w+)")
    ddr_finish_re = re.compile(r"\[DDR\]\s+finish request, send\s+(\w+)")
    ddr_finish_legacy_re = re.compile(r"\[DDR\]\s+finish processing current request")

    event_sync_block_re = re.compile(
        r"\[(SAInstQueue|SFUInstQueue)\]\s+Inst Issue Blocked by Event Sync Unit"
    )

    def __init__(self) -> None:
        self.current_cycle = 0
        self.last_cycle = 0

        self.intervals: List[Interval] = []
        self.markers: List[Marker] = []
        self.predictions: List[Prediction] = []

        self.pending_dma_ops: Deque[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = deque()
        self.pending_sa_words: List[Tuple[int, int]] = []
        self.pending_sa_resources_queue: Deque[Tuple[Tuple[str, ...], Tuple[str, ...]]] = deque()
        self.pending_acc_resources_queue: Deque[Tuple[Tuple[str, ...], Tuple[str, ...]]] = deque()
        self.blocked_sa_resources: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None

        self.open_dma: Dict[str, Tuple[int, str, Tuple[str, ...], Tuple[str, ...]]] = {}
        self.open_sa: Optional[Tuple[int, str, int, Tuple[str, ...], Tuple[str, ...]]] = None
        self.open_acc: Optional[Tuple[int, str, int, Tuple[str, ...], Tuple[str, ...]]] = None
        self.open_sfu: Optional[Tuple[int, str, int]] = None
        self.open_ddr: Optional[Tuple[int, str]] = None

        self.open_waits: Dict[str, int] = {}

    @staticmethod
    def _decode_sa_resources(
        *,
        is_acc: bool,
        is_bias: bool,
        acc_dest: bool,
        a_keep: bool,
        b_keep: bool,
        bias_keep: bool,
        a_addr: int,
        b_addr: int,
        biaspsum_addr: int,
        output_addr: int,
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        reads: List[str] = []
        if not a_keep:
            reads.append(make_resource("SPM", a_addr))
        if not b_keep:
            reads.append(make_resource("SPM", b_addr))
        if not bias_keep:
            if is_bias:
                reads.append(make_resource("ACC_BIAS", biaspsum_addr))
            elif is_acc:
                reads.append(make_resource("ACC", biaspsum_addr))

        output_space = "ACC" if acc_dest else "SPM"
        writes = [make_resource(output_space, output_addr)]
        return ordered_unique_resources(reads), ordered_unique_resources(writes)

    def _decode_sa_resources_from_words(
        self,
        words: List[Tuple[int, int]],
    ) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        if len(words) != 3:
            return None

        _, inst0_low = words[0]
        inst1_high, inst1_low = words[1]
        inst2_high, inst2_low = words[2]
        opcode = inst0_low & 0xFF

        if opcode != 0x03:
            return None

        return self._decode_sa_resources(
            is_acc=((inst1_low >> 6) & 0x1) != 0,
            is_bias=((inst1_low >> 11) & 0x1) != 0,
            acc_dest=((inst1_low >> 3) & 0x1) != 0,
            a_keep=((inst1_low >> 4) & 0x1) != 0,
            b_keep=((inst1_low >> 5) & 0x1) != 0,
            bias_keep=((inst1_low >> 12) & 0x1) != 0,
            a_addr=inst2_low & 0xFFFFFFFF,
            b_addr=inst2_high & 0xFFFFFFFF,
            biaspsum_addr=inst1_high & 0xFFFFFFFF,
            output_addr=(inst1_high >> 32) & 0xFFFFFFFF,
        )

    def _decode_matadd_resources_from_words(
        self,
        words: List[Tuple[int, int]],
    ) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        if len(words) != 3:
            return None

        inst0_high, inst0_low = words[0]
        opcode = inst0_low & 0xFF
        if opcode != 0x04:
            return None

        a_addr = (inst0_low >> 32) & 0xFFFFFFFF
        b_addr = inst0_high & 0xFFFFFFFF
        output_addr = (inst0_high >> 32) & 0xFFFFFFFF

        reads = ordered_unique_resources(
            [
                make_resource("ACC", a_addr),
                make_resource("ACC", b_addr),
            ]
        )
        writes = ordered_unique_resources([make_resource("SPM", output_addr)])
        return reads, writes

    def _close_interval(
        self,
        lane: str,
        start: int,
        label: str,
        end: int,
        kind: str = "exec",
        note: str = "",
        color: Optional[str] = None,
        reads: Tuple[str, ...] = (),
        writes: Tuple[str, ...] = (),
    ) -> None:
        if end <= start:
            end = start + 1
        self.intervals.append(
            Interval(
                lane=lane,
                start=start,
                end=end,
                label=label,
                color=color or pick_color(label),
                kind=kind,
                note=note,
                reads=reads,
                writes=writes,
            )
        )

    def _add_marker(self, lane: str, cycle: int, label: str, kind: str, color: str) -> None:
        self.markers.append(Marker(lane=lane, cycle=cycle, label=label, kind=kind, color=color))

    def _add_prediction(self, lane: str, start: int, remaining: int, label: str, color: str) -> None:
        self.predictions.append(
            Prediction(
                lane=lane,
                start=start,
                predicted_end=start + remaining,
                label=label,
                color=color,
            )
        )

    def _open_wait(self, lane: str) -> None:
        if lane not in self.open_waits:
            self.open_waits[lane] = self.current_cycle

    def _close_wait(self, lane: str) -> None:
        if lane in self.open_waits:
            start = self.open_waits.pop(lane)
            self._close_interval(
                lane=lane,
                start=start,
                end=self.current_cycle,
                label="EventSyncWait",
                color=pick_color("EventSyncWait"),
                kind="wait",
                note="Blocked by Event Sync Unit",
            )

    def parse(self, log_path: Path) -> Tuple[List[Interval], List[Marker], List[Prediction], int]:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")

                m = self.cycle_re.search(line)
                if m:
                    self.current_cycle = int(m.group(1))
                    self.last_cycle = max(self.last_cycle, self.current_cycle)
                    continue

                self.last_cycle = max(self.last_cycle, self.current_cycle)

                # ---------- Event Sync wait ----------
                m = self.sa_block_detail_re.search(line)
                if m:
                    opcode = int(m.group(1), 16)
                    is_acc = m.group(2) == "1"
                    is_bias = m.group(3) == "1"
                    acc_dest = m.group(4) == "1"
                    a_keep = m.group(5) == "1"
                    b_keep = m.group(6) == "1"
                    bias_keep = m.group(7) == "1"
                    a_addr = int(m.group(8), 16)
                    b_addr = int(m.group(9), 16)
                    biaspsum_addr = int(m.group(10), 16)
                    output_addr = int(m.group(11), 16)

                    if opcode == 0x03:
                        self.blocked_sa_resources = self._decode_sa_resources(
                            is_acc=is_acc,
                            is_bias=is_bias,
                            acc_dest=acc_dest,
                            a_keep=a_keep,
                            b_keep=b_keep,
                            bias_keep=bias_keep,
                            a_addr=a_addr,
                            b_addr=b_addr,
                            biaspsum_addr=biaspsum_addr,
                            output_addr=output_addr,
                        )
                    else:
                        self.blocked_sa_resources = None

                    self._open_wait("SA")
                    continue

                m = self.event_sync_block_re.search(line)
                if m:
                    queue_name = m.group(1)
                    if queue_name == "SAInstQueue":
                        self._open_wait("SA")
                    elif queue_name == "SFUInstQueue":
                        self._open_wait("SFU")
                    continue

                # ---------- DMA ----------
                m = self.dma_mvin_re.search(line)
                if m:
                    local_addr = int(m.group(2), 16)
                    dest_space = m.group(3)
                    self.pending_dma_ops.append(
                        ("MVIN", (), ordered_unique_resources([make_resource(dest_space, local_addr)]))
                    )
                    continue

                m = self.dma_mvout_re.search(line)
                if m:
                    local_addr = int(m.group(1), 16)
                    source_space = m.group(3)
                    self.pending_dma_ops.append(
                        ("MVOUT", ordered_unique_resources([make_resource(source_space, local_addr)]), ())
                    )
                    continue

                m = self.dma_op_re.search(line)
                if m:
                    self.pending_dma_ops.append((m.group(1), (), ()))
                    continue

                m = self.sa_queue_word_re.search(line)
                if m:
                    word = (int(m.group(1), 16), int(m.group(2), 16))
                    inst_count = int(m.group(3))
                    inst_count_upper = int(m.group(4))

                    if inst_count == 1:
                        self.pending_sa_words = [word]
                    else:
                        self.pending_sa_words.append(word)

                    if inst_count == inst_count_upper:
                        sa_decoded = self._decode_sa_resources_from_words(self.pending_sa_words)
                        if sa_decoded is not None:
                            self.pending_sa_resources_queue.append(sa_decoded)

                        acc_decoded = self._decode_matadd_resources_from_words(self.pending_sa_words)
                        if acc_decoded is not None:
                            self.pending_acc_resources_queue.append(acc_decoded)

                        self.pending_sa_words = []
                    continue

                m = self.dma_issue_re.search(line)
                if m:
                    dma_id = m.group(1)
                    lane = f"DMA{dma_id}"
                    label, reads, writes = self.pending_dma_ops.popleft() if self.pending_dma_ops else ("DMA", (), ())

                    if lane in self.open_dma:
                        old_start, old_label, old_reads, old_writes = self.open_dma.pop(lane)
                        self._close_interval(
                            lane,
                            old_start,
                            old_label,
                            self.current_cycle,
                            reads=old_reads,
                            writes=old_writes,
                        )

                    self.open_dma[lane] = (self.current_cycle, label, reads, writes)
                    self._add_marker(lane, self.current_cycle, label, "issue", pick_color(label))
                    continue

                m = self.dma_retire_re.search(line)
                if m:
                    dma_id = m.group(1)
                    lane = f"DMA{dma_id}"
                    if lane in self.open_dma:
                        start, label, reads, writes = self.open_dma.pop(lane)
                        self._close_interval(
                            lane,
                            start,
                            label,
                            self.current_cycle + 1,
                            reads=reads,
                            writes=writes,
                        )
                        self._add_marker(lane, self.current_cycle, label, "retire", pick_color(label))
                    continue

                # ---------- SA ----------
                m = self.sa_start_re.search(line)
                if m:
                    label = m.group(1)
                    remaining = int(m.group(2))

                    self._close_wait("SA")

                    if self.open_sa is not None:
                        start, old_label, _, old_reads, old_writes = self.open_sa
                        self._close_interval("SA", start, old_label, self.current_cycle, reads=old_reads, writes=old_writes)

                    if self.pending_sa_resources_queue:
                        reads, writes = self.pending_sa_resources_queue.popleft()
                    else:
                        reads, writes = self.blocked_sa_resources or ((), ())
                    self.blocked_sa_resources = None
                    self.open_sa = (self.current_cycle, label, remaining, reads, writes)
                    color = pick_color(label)
                    self._add_marker("SA", self.current_cycle, label, "start", color)
                    self._add_prediction("SA", self.current_cycle, remaining, label, color)
                    continue

                m = self.sa_issue_re.search(line)
                if m:
                    self._close_wait("SA")
                    continue

                m = self.sa_finish_re.search(line)
                if m:
                    if self.open_sa is not None:
                        start, label, _, reads, writes = self.open_sa
                        self._close_interval("SA", start, label, self.current_cycle + 1, reads=reads, writes=writes)
                        self._add_marker("SA", self.current_cycle, label, "finish", pick_color(label))
                        self.open_sa = None
                    continue

                # ---------- ACC ----------
                m = self.acc_start_re.search(line)
                if m:
                    label = m.group(1)
                    remaining = int(m.group(2))

                    if self.open_acc is not None:
                        start, old_label, _, old_reads, old_writes = self.open_acc
                        self._close_interval("ACC", start, old_label, self.current_cycle, reads=old_reads, writes=old_writes)

                    if label == "Matadd" and self.pending_acc_resources_queue:
                        reads, writes = self.pending_acc_resources_queue.popleft()
                    else:
                        reads, writes = (), ()

                    self.open_acc = (self.current_cycle, label, remaining, reads, writes)
                    color = pick_color(label)
                    self._add_marker("ACC", self.current_cycle, label, "start", color)
                    self._add_prediction("ACC", self.current_cycle, remaining, label, color)
                    continue

                m = self.sa_retire_re.search(line)
                if m:
                    if self.open_acc is not None:
                        start, label, _, reads, writes = self.open_acc
                        self._close_interval("ACC", start, label, self.current_cycle + 1, reads=reads, writes=writes)
                        self._add_marker("ACC", self.current_cycle, label, "retire", pick_color(label))
                        self.open_acc = None
                    continue

                # ---------- SFU ----------
                m = self.sfu_start_re.search(line)
                if m:
                    label = m.group(1)
                    remaining = int(m.group(2))

                    self._close_wait("SFU")

                    if self.open_sfu is not None:
                        start, old_label, _ = self.open_sfu
                        self._close_interval("SFU", start, old_label, self.current_cycle)

                    self.open_sfu = (self.current_cycle, label, remaining)
                    color = pick_color(label)
                    self._add_marker("SFU", self.current_cycle, label, "start", color)
                    self._add_prediction("SFU", self.current_cycle, remaining, label, color)
                    continue

                m = self.sfu_issue_re.search(line)
                if m:
                    self._close_wait("SFU")
                    continue

                m = self.sfu_retire_re.search(line)
                if m:
                    if self.open_sfu is not None:
                        start, label, _ = self.open_sfu
                        self._close_interval("SFU", start, label, self.current_cycle + 1)
                        self._add_marker("SFU", self.current_cycle, label, "retire", pick_color(label))
                        self.open_sfu = None
                    continue

                # ---------- DDR ----------
                m = self.ddr_start_re.search(line)
                if m:
                    label = m.group(1)

                    if self.open_ddr is not None:
                        start, old_label = self.open_ddr
                        self._close_interval("DDR", start, old_label, self.current_cycle)

                    self.open_ddr = (self.current_cycle, label)
                    self._add_marker("DDR", self.current_cycle, label, "start", pick_color(label))
                    continue

                m = self.ddr_finish_re.search(line)
                if m:
                    if self.open_ddr is not None:
                        start, label = self.open_ddr
                        self._close_interval("DDR", start, label, self.current_cycle + 1)
                        self._add_marker("DDR", self.current_cycle, label, "finish", pick_color(label))
                        self.open_ddr = None
                    continue

                m = self.ddr_finish_legacy_re.search(line)
                if m:
                    if self.open_ddr is not None:
                        start, label = self.open_ddr
                        self._close_interval("DDR", start, label, self.current_cycle + 1)
                        self._add_marker("DDR", self.current_cycle, label, "finish", pick_color(label))
                        self.open_ddr = None
                    continue

        end_cycle = self.last_cycle + 1

        for lane, (start, label, reads, writes) in list(self.open_dma.items()):
            self._close_interval(lane, start, label, end_cycle, reads=reads, writes=writes)

        if self.open_sa is not None:
            start, label, remaining, reads, writes = self.open_sa
            self._close_interval("SA", start, label, max(end_cycle, start + remaining), reads=reads, writes=writes)

        if self.open_acc is not None:
            start, label, remaining, reads, writes = self.open_acc
            self._close_interval("ACC", start, label, max(end_cycle, start + remaining), reads=reads, writes=writes)

        if self.open_sfu is not None:
            start, label, remaining = self.open_sfu
            self._close_interval("SFU", start, label, max(end_cycle, start + remaining))

        if self.open_ddr is not None:
            start, label = self.open_ddr
            self._close_interval("DDR", start, label, end_cycle)

        for lane, start in list(self.open_waits.items()):
            self._close_interval(
                lane=lane,
                start=start,
                end=end_cycle,
                label="EventSyncWait",
                color=pick_color("EventSyncWait"),
                kind="wait",
                note="Blocked by Event Sync Unit",
            )

        return self.intervals, self.markers, self.predictions, end_cycle


# ==========================================================
# Post-process
# ==========================================================
def filter_intervals(intervals: List[Interval], start_cycle: int, end_cycle: int) -> List[Interval]:
    filtered = []
    for iv in intervals:
        s = max(iv.start, start_cycle)
        e = min(iv.end, end_cycle)
        if e > s:
            filtered.append(
                Interval(
                    lane=iv.lane,
                    start=s,
                    end=e,
                    label=iv.label,
                    color=iv.color,
                    kind=iv.kind,
                    note=iv.note,
                    reads=iv.reads,
                    writes=iv.writes,
                )
            )
    return filtered


def filter_markers(markers: List[Marker], start_cycle: int, end_cycle: int) -> List[Marker]:
    return [m for m in markers if start_cycle <= m.cycle <= end_cycle]


def filter_predictions(predictions: List[Prediction], start_cycle: int, end_cycle: int) -> List[Prediction]:
    result = []
    for p in predictions:
        if p.predicted_end < start_cycle:
            continue
        if p.start > end_cycle:
            continue
        result.append(p)
    return result


def derive_dependencies(intervals: List[Interval], max_per_target: int = 3) -> List[Dependency]:
    exec_intervals = [iv for iv in intervals if iv.kind == "exec"]
    waits_by_lane: Dict[str, List[Interval]] = defaultdict(list)
    exec_by_lane: Dict[str, List[Interval]] = defaultdict(list)
    dependency_lanes = {iv.lane for iv in exec_intervals if iv.lane.startswith("DMA")} | {"SA", "SFU"}

    for iv in intervals:
        if iv.kind == "wait":
            waits_by_lane[iv.lane].append(iv)
        else:
            exec_by_lane[iv.lane].append(iv)

    for lane in waits_by_lane:
        waits_by_lane[lane].sort(key=lambda x: (x.start, x.end))
    for lane in exec_by_lane:
        exec_by_lane[lane].sort(key=lambda x: (x.start, x.end))

    deps: List[Dependency] = []

    def find_preceding_wait(lane: str, start_cycle: int) -> Optional[Interval]:
        candidates = waits_by_lane.get(lane, [])
        best = None
        for iv in candidates:
            if iv.end <= start_cycle and start_cycle - iv.end <= 1:
                best = iv
        return best

    def find_address_matched_sources(target: Interval) -> List[Interval]:
        matched: Dict[Tuple[str, int, int, str], Interval] = {}
        if not target.reads:
            return []

        for resource in target.reads:
            best = None
            for src in exec_intervals:
                if src is target:
                    continue
                if src.lane not in dependency_lanes:
                    continue
                if not src.writes or resource not in src.writes:
                    continue
                if src.end <= target.start and (best is None or src.end > best.end):
                    best = src
            if best is not None:
                matched[(best.lane, best.start, best.end, best.label)] = best

        return sorted(matched.values(), key=lambda x: x.end)

    # SA / SFU: prefer exact resource matches; fall back to wait-window timing when metadata is missing
    for lane in ["SA", "SFU"]:
        for target in exec_by_lane.get(lane, []):
            picked = find_address_matched_sources(target)

            if not picked:
                wait = find_preceding_wait(lane, target.start)
                if wait is None:
                    continue

                latest_by_lane: Dict[str, Interval] = {}
                for src in exec_intervals:
                    if src.lane == lane:
                        continue
                    if src.lane not in dependency_lanes:
                        continue
                    if wait.start <= src.end <= target.start:
                        latest_by_lane[src.lane] = src

                picked = sorted(latest_by_lane.values(), key=lambda x: x.end, reverse=True)
                picked = list(reversed(picked[:max_per_target]))
            elif max_per_target > 0 and len(picked) > max_per_target:
                picked = picked[-max_per_target:]

            for src in picked:
                deps.append(
                    Dependency(
                        src_lane=src.lane,
                        src_cycle=src.end,
                        src_label=src.label,
                        dst_lane=target.lane,
                        dst_cycle=target.start,
                        dst_label=target.label,
                        label=f"{src.lane}:{src.label}",
                        color=pick_color("DEP"),
                    )
                )

    # ACC: only Matadd needs dependency arrows
    for target in exec_by_lane.get("ACC", []):
        if target.label != "Matadd":
            continue

        picked = find_address_matched_sources(target)
        if max_per_target > 0 and len(picked) > max_per_target:
            picked = picked[-max_per_target:]

        for src in picked:
            deps.append(
                Dependency(
                    src_lane=src.lane,
                    src_cycle=src.end,
                    src_label=src.label,
                    dst_lane=target.lane,
                    dst_cycle=target.start,
                    dst_label=target.label,
                    label=f"{src.lane}:{src.label}",
                    color=pick_color("DEP"),
                )
            )

    return deps


# ==========================================================
# Plot
# ==========================================================
def build_plotly_figure(
    intervals: List[Interval],
    markers: List[Marker],
    predictions: List[Prediction],
    dependencies: List[Dependency],
    title: str,
    lanes: List[str],
    show_markers: bool,
    show_predictions: bool,
    show_waits: bool,
    show_dependencies: bool,
    show_dependency_labels: bool,
    only_active_lanes: bool,
) -> go.Figure:
    lane_to_intervals: Dict[str, List[Interval]] = defaultdict(list)
    lane_to_markers: Dict[str, List[Marker]] = defaultdict(list)

    for iv in intervals:
        lane_to_intervals[iv.lane].append(iv)
    for mk in markers:
        lane_to_markers[mk.lane].append(mk)

    if only_active_lanes:
        active_lanes = [lane for lane in lanes if lane_to_intervals.get(lane) or lane_to_markers.get(lane)]
    else:
        active_lanes = lanes[:]

    if not active_lanes:
        raise RuntimeError("No timeline events found in the selected cycle range.")

    y_pos = {lane: len(active_lanes) - 1 - i for i, lane in enumerate(active_lanes)}

    visible_intervals = []
    for iv in intervals:
        if iv.lane not in y_pos:
            continue
        if iv.kind == "wait" and not show_waits:
            continue
        visible_intervals.append(iv)

    xmin = min(iv.start for iv in visible_intervals) if visible_intervals else 0
    xmax = max(iv.end for iv in visible_intervals) if visible_intervals else 1

    fig = go.Figure()
    bar_h = 0.50
    label_thresholds = {"exec": 18, "wait": 24}

    # lane bands + baseline
    for lane_index, lane in enumerate(active_lanes):
        y = y_pos[lane]
        fig.add_shape(
            type="rect",
            x0=xmin,
            x1=xmax,
            y0=y - 0.56,
            y1=y + 0.56,
            line=dict(width=0),
            fillcolor=lane_band_color(lane_index),
            layer="below",
        )
        fig.add_shape(
            type="line",
            x0=xmin,
            x1=xmax,
            y0=y,
            y1=y,
            line=dict(color="#CBD5E1", width=2.8),
            layer="below",
        )

    # one trace per interval => hover时可以单独加粗边框
    for iv in visible_intervals:
        y = y_pos[iv.lane]
        if iv.kind == "wait":
            fillcolor = "rgba(216,211,203,0.38)"
            line_color = "#A8A29E"
            pattern_shape = "/"
        else:
            fillcolor = hex_to_rgba(iv.color, 0.72)
            line_color = iv.color
            pattern_shape = ""

        width = iv.end - iv.start
        hover_text = (
            f"lane={iv.lane}<br>"
            f"label={iv.label}<br>"
            f"start={iv.start}<br>"
            f"end={iv.end}<br>"
            f"duration={width}"
        )
        if iv.note:
            hover_text += f"<br>note={iv.note}"

        show_interval_label = width >= label_thresholds[iv.kind]

        fig.add_trace(
            go.Bar(
                x=[width],
                y=[y],
                base=[iv.start],
                orientation="h",
                width=bar_h,
                marker=dict(
                    color=fillcolor,
                    line=dict(color=line_color, width=1.5),
                    pattern=dict(shape=pattern_shape, solidity=0.16 if iv.kind == "wait" else 0.0),
                ),
                hovertemplate=hover_text + "<extra></extra>",
                name=iv.label,
                text=[iv.label] if show_interval_label else None,
                textposition="inside",
                textfont=dict(size=10, color="#0F172A"),
                insidetextanchor="middle",
                constraintext="inside",
                cliponaxis=False,
                showlegend=False,
                meta={
                    "kind": "interval",
                    "base_width": 1.5,
                    "hover_width": 3.2,
                    "interval_kind": iv.kind,
                },
            )
        )

    # markers
    if show_markers:
        symbol_map = {
            "start": "triangle-up",
            "finish": "triangle-down",
            "issue": "circle",
            "retire": "square",
        }

        grouped: Dict[str, List[Marker]] = defaultdict(list)
        for mk in markers:
            if mk.lane in y_pos:
                grouped[mk.kind].append(mk)

        for kind, items in grouped.items():
            fig.add_trace(
                go.Scatter(
                    x=[m.cycle for m in items],
                    y=[y_pos[m.lane] for m in items],
                    mode="markers",
                    marker=dict(
                        symbol=symbol_map.get(kind, "circle"),
                        size=9,
                        color=[m.color for m in items],
                        line=dict(color="black", width=0.8),
                    ),
                    text=[f"{m.lane} | {m.label} | {kind}" for m in items],
                    hovertemplate="marker=%{text}<br>cycle=%{x}<extra></extra>",
                    name=f"marker:{kind}",
                    showlegend=False,
                )
            )

    # prediction lines
    if show_predictions:
        for pr in predictions:
            if pr.lane not in y_pos:
                continue
            y = y_pos[pr.lane]

            fig.add_shape(
                type="line",
                x0=pr.predicted_end,
                x1=pr.predicted_end,
                y0=y - 0.42,
                y1=y + 0.42,
                line=dict(color=pr.color, width=1.3, dash="dash"),
                opacity=0.95,
            )

            fig.add_shape(
                type="line",
                x0=pr.start,
                x1=pr.predicted_end,
                y0=y + 0.39,
                y1=y + 0.39,
                line=dict(color=pr.color, width=1.0, dash="dot"),
                opacity=0.75,
            )

    # dependencies
    if show_dependencies:
        route_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for dep in dependencies:
            if dep.src_lane not in y_pos or dep.dst_lane not in y_pos:
                continue

            route_key = (dep.src_lane, dep.dst_lane)
            route_index = route_counts[route_key]
            route_counts[route_key] += 1
            start, ctrl, end, label_pt, arrow_tail = dependency_curve_geometry(dep, y_pos, route_index)
            dep_color = hex_to_rgba(dep.color, 0.70)

            fig.add_shape(
                type="path",
                path=(
                    f"M {start[0]},{start[1]} "
                    f"Q {ctrl[0]},{ctrl[1]} {end[0]},{end[1]}"
                ),
                line=dict(color=dep_color, width=1.5),
                layer="above",
            )

            # 只在终点附近保留短箭头，避免整条依赖线压在矩形区域上
            fig.add_annotation(
                x=end[0],
                y=end[1],
                ax=arrow_tail[0],
                ay=arrow_tail[1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=3,
                arrowsize=0.9,
                arrowwidth=1.5,
                arrowcolor=dep_color,
            )

            if show_dependency_labels:
                fig.add_annotation(
                    x=label_pt[0],
                    y=label_pt[1] + 0.08,
                    text=dep.label,
                    showarrow=False,
                    font=dict(size=9, color=dep.color),
                    bgcolor="rgba(255,255,255,0.84)",
                    bordercolor=hex_to_rgba(dep.color, 0.55),
                    borderwidth=0.6,
                    borderpad=2,
                )

    # custom legend
    legend_labels = []
    for iv in visible_intervals:
        if iv.kind == "wait":
            continue
        if iv.label not in legend_labels:
            legend_labels.append(iv.label)

    for label in legend_labels:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=pick_color(label), width=8),
                name=label,
                showlegend=True,
                hoverinfo="skip",
            )
        )

    if show_waits:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=pick_color("EventSyncWait"), width=8, dash="dot"),
                name="EventSyncWait",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    if show_predictions:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=pick_color("PRED"), width=2, dash="dash"),
                name="Predicted End",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    if show_dependencies:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=pick_color("DEP"), width=2),
                name="Dependency",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        plot_bgcolor="#FBF8F2",
        paper_bgcolor="#FFFDF8",
        barmode="overlay",
        xaxis=dict(
            title="Cycle",
            showgrid=True,
            gridcolor="#E7E2D8",
            zeroline=False,
        ),
        yaxis=dict(
            title="Module",
            tickmode="array",
            tickvals=[y_pos[l] for l in active_lanes],
            ticktext=active_lanes,
            range=[-1.15, len(active_lanes) - 0.15 + 1.65],
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=80, r=50, t=90, b=120),
        height=max(460, 128 * len(active_lanes)),
        hovermode="closest",
        font=dict(family="Helvetica, Arial, sans-serif", color="#334155"),
    )

    return fig


# ==========================================================
# HTML Hover Enhancement
# ==========================================================
def save_html_with_hover_highlight(fig: go.Figure, out_html: Path, include_plotlyjs="cdn") -> None:
    html = pio.to_html(fig, full_html=True, include_plotlyjs=include_plotlyjs)

    extra_js = r"""
<script>
document.addEventListener('DOMContentLoaded', function () {
    const gd = document.querySelector('.plotly-graph-div');
    if (!gd) return;

    let hoveredTrace = null;

    function roundedRectPath(x, y, w, h, r) {
        return [
            `M ${x + r} ${y}`,
            `H ${x + w - r}`,
            `Q ${x + w} ${y} ${x + w} ${y + r}`,
            `V ${y + h - r}`,
            `Q ${x + w} ${y + h} ${x + w - r} ${y + h}`,
            `H ${x + r}`,
            `Q ${x} ${y + h} ${x} ${y + h - r}`,
            `V ${y + r}`,
            `Q ${x} ${y} ${x + r} ${y}`,
            'Z'
        ].join(' ');
    }

    function applyRoundedBars() {
        const paths = gd.querySelectorAll('.barlayer .trace path');
        paths.forEach(function (p) {
            const bb = p.getBBox();
            if (!bb || bb.width <= 0 || bb.height <= 0) return;
            const r = Math.min(11, bb.height / 2.0, bb.width / 6.0);
            p.setAttribute('d', roundedRectPath(bb.x, bb.y, bb.width, bb.height, r));
            p.style.strokeLinejoin = 'round';
            p.style.strokeLinecap = 'round';
        });
    }

    function restoreTrace(idx) {
        if (idx === null || idx === undefined) return;
        const tr = gd.data[idx];
        if (!tr || !tr.meta || tr.meta.kind !== 'interval') return;
        const baseWidth = (tr.meta.base_width !== undefined) ? tr.meta.base_width : 1.4;
        Plotly.restyle(gd, {'marker.line.width': baseWidth}, [idx]).then(() => {
            requestAnimationFrame(applyRoundedBars);
        });
    }

    gd.on('plotly_hover', function (ev) {
        if (!ev || !ev.points || !ev.points.length) return;

        const idx = ev.points[0].curveNumber;
        const tr = gd.data[idx];
        if (!tr || !tr.meta || tr.meta.kind !== 'interval') return;

        if (hoveredTrace !== null && hoveredTrace !== idx) {
            restoreTrace(hoveredTrace);
        }

        const hoverWidth = (tr.meta.hover_width !== undefined) ? tr.meta.hover_width : 3.2;
        Plotly.restyle(gd, {'marker.line.width': hoverWidth}, [idx]).then(() => {
            requestAnimationFrame(applyRoundedBars);
        });
        hoveredTrace = idx;
    });

    gd.on('plotly_unhover', function () {
        restoreTrace(hoveredTrace);
        hoveredTrace = null;
    });

    gd.on('plotly_afterplot', function () {
        requestAnimationFrame(applyRoundedBars);
    });

    requestAnimationFrame(applyRoundedBars);
});
</script>
"""

    html = html.replace("</body>", extra_js + "\n</body>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


# ==========================================================
# Main
# ==========================================================
def main() -> None:
    log_path = Path(CONFIG["log"])
    out_html = Path(CONFIG["out_html"])
    out_png = Path(CONFIG["out_png"]) if CONFIG["out_png"] else None

    parser = TimelineParser()
    intervals, markers, predictions, last_cycle = parser.parse(log_path)

    start_cycle = int(CONFIG["start_cycle"])
    end_cycle_cfg = CONFIG["end_cycle"]
    end_cycle = last_cycle if end_cycle_cfg is None else int(end_cycle_cfg)

    if end_cycle <= start_cycle:
        raise ValueError("CONFIG['end_cycle'] must be greater than CONFIG['start_cycle'].")

    intervals = filter_intervals(intervals, start_cycle, end_cycle)
    markers = filter_markers(markers, start_cycle, end_cycle)
    predictions = filter_predictions(predictions, start_cycle, end_cycle)

    dependencies = derive_dependencies(
        intervals,
        max_per_target=int(CONFIG["max_dependency_per_target"]),
    )

    fig = build_plotly_figure(
        intervals=intervals,
        markers=markers,
        predictions=predictions,
        dependencies=dependencies,
        title=CONFIG["title"],
        lanes=list(CONFIG["lanes"]),
        show_markers=bool(CONFIG["show_markers"]),
        show_predictions=bool(CONFIG["show_predictions"]),
        show_waits=bool(CONFIG["show_waits"]),
        show_dependencies=bool(CONFIG["show_dependencies"]),
        show_dependency_labels=bool(CONFIG["show_dependency_labels"]),
        only_active_lanes=bool(CONFIG["only_active_lanes"]),
    )

    save_html_with_hover_highlight(
        fig,
        out_html=out_html,
        include_plotlyjs=CONFIG["html_include_plotlyjs"],
    )
    print(f"[OK] Saved interactive HTML: {out_html}")

    if out_png is not None:
        try:
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(out_png, scale=2)
            print(f"[OK] Saved PNG: {out_png}")
        except Exception as e:
            print(f"[WARN] PNG export skipped: {e}")
            print("[WARN] If you need PNG, install kaleido: pip install kaleido")


if __name__ == "__main__":
    main()
