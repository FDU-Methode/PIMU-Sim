#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import re
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent


# ==========================================================
# User Config
# 只需要改这里，然后直接运行：python3 plot_dma_dispatch_matrix.py
# ==========================================================
CONFIG = {
    # 二选一：
    # 1) 用日志直接解析 DMA dispatch
    "log": BASE_DIR / "log/tb_resnet_block_log.txt",
    # 2) 或者直接读取已有 CSV。若不为 None，则优先使用它
    "csv": None,

    # 可选：当从 log 解析时，是否顺手导出 CSV
    "csv_out": None,

    # 输出 SVG；若为 None，则按输入文件名自动推导
    "out": BASE_DIR / "log/tb_resnet_block_dma_dispatch_matrix.svg",

    "start_req": 30,
    "max_reqs": 30,
    "title": "PIMU-Sim Multi-DMA Dispatch Matrix",
}


@dataclass
class DmaReq:
    index: int
    lane: str
    op: str
    start: int
    end: int
    duration: int
    src: str
    dst: str
    rows: str
    cols: str
    local_space: str


@dataclass
class DmaInterval:
    lane: str
    op: str
    start: int
    end: int
    src: str
    dst: str
    rows: str
    cols: str
    local_space: str


@dataclass
class PendingDmaOp:
    op: str
    src: str
    dst: str
    rows: str
    cols: str
    local_space: str


CYCLE_RE = re.compile(r"\[Core Cycle:\s*(\d+)\]")
DMA_MVIN_RE = re.compile(
    r"\[DMA\]\s+MVIN:\s+send\s+LD_REQ,\s+src=0x([0-9a-fA-F]+)\s+dst=0x([0-9a-fA-F]+)\s+"
    r"rows=(\d+)\s+cols=(\d+).*dest=(SPM|ACC)"
)
DMA_MVOUT_RE = re.compile(
    r"\[DMA\]\s+MVOUT:\s+send\s+ST_REQ,\s+src=0x([0-9a-fA-F]+)\s+dst=0x([0-9a-fA-F]+)\s+"
    r"rows=(\d+)\s+cols=(\d+).*source=(SPM|ACC)"
)
DMA_ISSUE_RE = re.compile(r"\[Core\]\s+DMA\s+(\d)\s+Queue Issued")
DMA_RETIRE_RE = re.compile(r"\[Core\]\s+DMA\s+(\d)\s+Instruction Retired")


def parse_dma_log(log_path: Path) -> Tuple[List[DmaInterval], int, int]:
    pending_ops: Deque[PendingDmaOp] = deque()
    open_ops: Dict[str, Tuple[int, PendingDmaOp]] = {}
    intervals: List[DmaInterval] = []
    current_cycle = 0
    last_cycle = 0

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            m = CYCLE_RE.search(line)
            if m:
                current_cycle = int(m.group(1))
                last_cycle = max(last_cycle, current_cycle)
                continue

            last_cycle = max(last_cycle, current_cycle)

            m = DMA_MVIN_RE.search(line)
            if m:
                pending_ops.append(
                    PendingDmaOp(
                        op="MVIN",
                        src=f"0x{m.group(1).lower()}",
                        dst=f"0x{m.group(2).lower()}",
                        rows=m.group(3),
                        cols=m.group(4),
                        local_space=m.group(5),
                    )
                )
                continue

            m = DMA_MVOUT_RE.search(line)
            if m:
                pending_ops.append(
                    PendingDmaOp(
                        op="MVOUT",
                        src=f"0x{m.group(1).lower()}",
                        dst=f"0x{m.group(2).lower()}",
                        rows=m.group(3),
                        cols=m.group(4),
                        local_space=m.group(5),
                    )
                )
                continue

            m = DMA_ISSUE_RE.search(line)
            if m:
                lane = f"DMA{m.group(1)}"
                if not pending_ops:
                    continue
                pending = pending_ops.popleft()
                open_ops[lane] = (current_cycle, pending)
                continue

            m = DMA_RETIRE_RE.search(line)
            if m:
                lane = f"DMA{m.group(1)}"
                if lane not in open_ops:
                    continue
                start_cycle, pending = open_ops.pop(lane)
                intervals.append(
                    DmaInterval(
                        lane=lane,
                        op=pending.op,
                        start=start_cycle,
                        end=current_cycle + 1,
                        src=pending.src,
                        dst=pending.dst,
                        rows=pending.rows,
                        cols=pending.cols,
                        local_space=pending.local_space,
                    )
                )

    for lane, (start_cycle, pending) in open_ops.items():
        intervals.append(
            DmaInterval(
                lane=lane,
                op=pending.op,
                start=start_cycle,
                end=last_cycle + 1,
                src=pending.src,
                dst=pending.dst,
                rows=pending.rows,
                cols=pending.cols,
                local_space=pending.local_space,
            )
        )

    intervals.sort(key=lambda item: (item.start, item.lane, item.op))
    return intervals, 0, last_cycle + 1


def write_csv(intervals: List[DmaInterval], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lane",
                "op",
                "start_cycle",
                "end_cycle",
                "duration",
                "src",
                "dst",
                "rows",
                "cols",
                "local_space",
            ],
        )
        writer.writeheader()
        for item in intervals:
            writer.writerow(
                {
                    "lane": item.lane,
                    "op": item.op,
                    "start_cycle": item.start,
                    "end_cycle": item.end,
                    "duration": item.end - item.start,
                    "src": item.src,
                    "dst": item.dst,
                    "rows": item.rows,
                    "cols": item.cols,
                    "local_space": item.local_space,
                }
            )


def read_reqs(csv_path: Path, start_req: int, max_reqs: Optional[int]) -> List[DmaReq]:
    reqs: List[DmaReq] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            if i < start_req:
                continue
            reqs.append(
                DmaReq(
                    index=i,
                    lane=row["lane"],
                    op=row["op"],
                    start=int(row["start_cycle"]),
                    end=int(row["end_cycle"]),
                    duration=int(row["duration"]),
                    src=row.get("src", ""),
                    dst=row.get("dst", ""),
                    rows=row.get("rows", ""),
                    cols=row.get("cols", ""),
                    local_space=row.get("local_space", ""),
                )
            )
            if max_reqs is not None and len(reqs) >= max_reqs:
                break
    return reqs


def reqs_from_log(log_path: Path, start_req: int, max_reqs: Optional[int], csv_out: Optional[Path]) -> List[DmaReq]:
    intervals, _, _ = parse_dma_log(log_path)
    if not intervals:
        raise RuntimeError(f"No DMA issue/retire intervals found in {log_path}")

    if csv_out is not None:
        write_csv(intervals, csv_out)

    reqs: List[DmaReq] = []
    for i, item in enumerate(intervals, start=1):
        if i < start_req:
            continue
        reqs.append(
            DmaReq(
                index=i,
                lane=item.lane,
                op=item.op,
                start=item.start,
                end=item.end,
                duration=item.end - item.start,
                src=item.src,
                dst=item.dst,
                rows=item.rows,
                cols=item.cols,
                local_space=item.local_space,
            )
        )
        if max_reqs is not None and len(reqs) >= max_reqs:
            break
    return reqs


def contiguous_spans(reqs: List[DmaReq]) -> List[Tuple[int, int, str, str]]:
    spans: List[Tuple[int, int, str, str]] = []
    if not reqs:
        return spans

    def phase(req: DmaReq) -> Tuple[str, str]:
        if req.op == "MVIN":
            return ("MVIN prefetch", "#7B1FA2")
        if req.op == "MVOUT":
            return ("MVOUT writeback", "#1565C0")
        return ("DMA", "#4B5563")

    start = 0
    current_label, current_color = phase(reqs[0])
    for i, req in enumerate(reqs[1:], start=1):
        label, color = phase(req)
        if label != current_label:
            spans.append((start, i - 1, current_label, current_color))
            start = i
            current_label, current_color = label, color
    spans.append((start, len(reqs) - 1, current_label, current_color))
    return spans


def render_matrix(reqs: List[DmaReq], out_path: Path, title: str) -> None:
    if not reqs:
        raise RuntimeError("No DMA requests found")

    lanes = ["DMA1", "DMA2", "DMA3"]
    lane_to_row = {lane: i for i, lane in enumerate(lanes)}

    cell_w = 104
    cell_h = 60
    left = 126
    top = 112
    right = 42
    bottom = 46
    width = left + len(reqs) * cell_w + right
    height = top + len(lanes) * cell_h + bottom

    base_fill = {
        "DMA1": "#ECE9F7",
        "DMA2": "#E6F0FF",
        "DMA3": "#E6F0FF",
    }
    active_fill = {
        ("MVIN", "ACC"): "#C9BFE5",
        ("MVIN", "SPM"): "#CFE1F9",
        ("MVOUT", "SPM"): "#F7D7B5",
        ("MVOUT", "ACC"): "#F0C7D5",
    }
    stroke = "#5A5B7A"

    def cell_x(col: int) -> int:
        return left + col * cell_w

    def cell_y(row: int) -> int:
        return top + row * cell_h

    def req_fill(req: DmaReq) -> str:
        return active_fill.get((req.op, req.local_space), "#D9DEE8")

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(
        "<defs>"
        "<marker id=\"arrow-blue\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\">"
        "<path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#0B63B6\"/>"
        "</marker>"
        "<marker id=\"arrow-purple\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\">"
        "<path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#7B1FA2\"/>"
        "</marker>"
        "<marker id=\"arrow-orange\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\">"
        "<path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#C46A1A\"/>"
        "</marker>"
        "</defs>"
    )
    svg.append("<style>")
    svg.append(
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}"
        ".title{font-size:20px;font-weight:700}"
        ".lane{font-size:16px;font-weight:700}"
        ".cell{font-size:15px;font-weight:700}"
        ".small{font-size:11px;fill:#374151}"
        ".phase{font-size:16px;font-style:italic;font-weight:700}"
    )
    svg.append("</style>")
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="{left}" y="32" class="title">{html.escape(title)}</text>')
    svg.append(f'<text x="{left}" y="54" class="small">Each column is one DMA instruction from the workload; highlighted cell shows selected DMA queue.</text>')
    svg.append(f'<text x="{left}" y="70" class="small">Policy shown here: ACC MVIN/MVOUT fixed to DMA1; SPM MVIN balanced across DMA2/3; SPM MVOUT may borrow DMA1.</text>')

    for begin, end, label, color in contiguous_spans(reqs):
        x1 = cell_x(begin)
        x2 = cell_x(end) + cell_w
        y = top - 48
        svg.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#111827" stroke-width="2"/>')
        svg.append(f'<line x1="{x1}" y1="{y - 9}" x2="{x1}" y2="{y + 9}" stroke="#111827" stroke-width="2"/>')
        svg.append(f'<line x1="{x2}" y1="{y - 9}" x2="{x2}" y2="{y + 9}" stroke="#111827" stroke-width="2"/>')
        svg.append(f'<text x="{(x1 + x2) / 2}" y="{y - 10}" text-anchor="middle" class="phase" fill="{color}">{html.escape(label)}</text>')

    for row, lane in enumerate(lanes):
        y = cell_y(row)
        svg.append(f'<text x="{left - 18}" y="{y + cell_h / 2 + 6}" text-anchor="end" class="lane">{lane}</text>')

    for col, req in enumerate(reqs):
        x = cell_x(col)
        svg.append(f'<text x="{x + cell_w / 2}" y="{top - 18}" text-anchor="middle" class="small">req {req.index}</text>')
        for row, lane in enumerate(lanes):
            y = cell_y(row)
            fill = req_fill(req) if lane == req.lane else base_fill[lane]
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="{stroke}" stroke-width="1.1"/>')
            if lane == req.lane:
                tooltip = (
                    f"req {req.index} -> {req.lane}\\n"
                    f"{req.op} {req.local_space}\\n"
                    f"cycle {req.start}-{req.end}, duration {req.duration}\\n"
                    f"src {req.src}, dst {req.dst}, rows {req.rows}, cols {req.cols}"
                )
                svg.append(f'<g><title>{html.escape(tooltip)}</title>')
                svg.append(f'<text x="{x + cell_w / 2}" y="{y + 24}" text-anchor="middle" class="cell">{req.op}</text>')
                svg.append(f'<text x="{x + cell_w / 2}" y="{y + 43}" text-anchor="middle" class="small">{html.escape(req.local_space)}  {req.start}-{req.end}</text>')
                svg.append("</g>")

    def add_arrow(req_idx: int, text: str, color: str, marker: str, x_shift: int = 0) -> None:
        if req_idx < 1 or req_idx > len(reqs):
            return
        req = reqs[req_idx - 1]
        row = lane_to_row.get(req.lane)
        if row is None:
            return
        x = cell_x(req_idx - 1) + cell_w / 2 + x_shift
        y1 = top - 6
        y2 = cell_y(row) + cell_h / 2 - 8
        svg.append(f'<text x="{x + 7}" y="{y1 + 8}" fill="{color}" font-size="12" font-weight="700">{html.escape(text)}</text>')
        svg.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="2.2" marker-end="url(#{marker})"/>')

    first_acc = next((req.index for req in reqs if req.local_space == "ACC"), None)
    first_spm_mvin = next((req.index for req in reqs if req.op == "MVIN" and req.local_space == "SPM"), None)
    first_mvout = next((req.index for req in reqs if req.op == "MVOUT"), None)
    if first_spm_mvin is not None:
        add_arrow(first_spm_mvin, "MVIN RR", "#0B63B6", "arrow-blue", x_shift=-20)
    if first_acc is not None:
        add_arrow(first_acc, "ACC -> DMA1", "#7B1FA2", "arrow-purple", x_shift=-4)
    if first_mvout is not None:
        add_arrow(first_mvout, "MVOUT helper", "#C46A1A", "arrow-orange", x_shift=-12)

    legend_x = left
    legend_y = height - 22
    legend = [
        ("MVIN to SPM", active_fill[("MVIN", "SPM")]),
        ("MVIN to ACC", active_fill[("MVIN", "ACC")]),
        ("MVOUT", active_fill[("MVOUT", "SPM")]),
    ]
    for i, (name, color) in enumerate(legend):
        x = legend_x + i * 142
        svg.append(f'<rect x="{x}" y="{legend_y - 12}" width="20" height="12" fill="{color}" stroke="{stroke}" stroke-width="0.8"/>')
        svg.append(f'<text x="{x + 28}" y="{legend_y - 2}" class="small">{html.escape(name)}</text>')

    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def resolve_path(path_like: Optional[Path | str]) -> Optional[Path]:
    if path_like is None:
        return None
    path = Path(path_like)
    return path if path.is_absolute() else BASE_DIR / path


def main() -> None:
    csv_path = resolve_path(CONFIG.get("csv"))
    log_path = resolve_path(CONFIG.get("log"))
    csv_out = resolve_path(CONFIG.get("csv_out"))
    out_path = resolve_path(CONFIG.get("out"))
    start_req = int(CONFIG.get("start_req", 1))
    max_reqs = CONFIG.get("max_reqs", 12)
    title = str(CONFIG.get("title", "PIMU-Sim Multi-DMA Dispatch Matrix"))

    if csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        reqs = read_reqs(csv_path, start_req, max_reqs)
        default_out = csv_path.with_name(csv_path.stem.replace("_schedule", "_dispatch_matrix") + ".svg")
    else:
        if log_path is None:
            raise ValueError("CONFIG['log'] and CONFIG['csv'] cannot both be None")
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        reqs = reqs_from_log(log_path, start_req, max_reqs, csv_out)
        default_out = log_path.with_name(log_path.stem.replace("_log", "") + "_dma_dispatch_matrix.svg")

    final_out = out_path if out_path is not None else default_out
    render_matrix(reqs, final_out, title)
    print(f"Wrote {final_out}")


if __name__ == "__main__":
    main()
