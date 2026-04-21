#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import plotly.graph_objects as go


# ==========================================================
# User Config
# 只需要改这里，然后直接运行：python3 plot_timeline.py
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "log": BASE_DIR / "log/tb_resnet_block_log.txt",
    "out_html": BASE_DIR / "log/tb_resnet_block_timeline.html",
    "out_png": None,  # 例如：BASE_DIR / "log/tb_resnet_block_timeline.png"
    "title": "PIMU-Sim Execution Timeline",
    "start_cycle": 0,
    "end_cycle": 5000,   # None 表示直到日志结束
    "lanes": ["DMA1", "DMA2", "DMA3", "SA", "ACC", "SFU", "DDR"],
    "show_markers": True,
    "show_predictions": True,
    "show_waits": True,
    "only_active_lanes": True,
    "html_include_plotlyjs": "cdn",  # "cdn" / True
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
    kind: str = "exec"     # exec / wait
    note: str = ""


@dataclass
class Marker:
    lane: str
    cycle: int
    label: str
    kind: str              # start / finish / issue / retire
    color: str


@dataclass
class Prediction:
    lane: str
    start: int
    predicted_end: int
    label: str
    color: str


COLOR_MAP = {
    "MVIN": "#4C78A8",
    "MVOUT": "#F58518",
    "GEMM": "#54A24B",
    "CONV": "#E45756",
    "Matadd": "#B279A2",
    "Softmax": "#72B7B2",
    "Layernorm": "#9C755F",
    "Gelu": "#FF9DA6",
    "Transpose": "#BAB0AC",
    "Pool_max": "#EECA3B",
    "Resample": "#8E6CFF",
    "LD_REQ": "#4C78A8",
    "ST_REQ": "#F58518",
    "DDR": "#9D9DA1",
    "WAIT": "#CFCFCF",
    "PRED": "#444444",
}


def pick_color(label: str, default: str = "#999999") -> str:
    return COLOR_MAP.get(label, default)


# ==========================================================
# Parser
# ==========================================================
class TimelineParser:
    cycle_re = re.compile(r"\[Core Cycle:\s*(\d+)\]")

    dma_op_re = re.compile(r"\[DMA\]\s+(MVIN|MVOUT):")
    dma_issue_re = re.compile(r"\[Core\]\s+DMA\s+(\d)\s+Queue Issued")
    dma_retire_re = re.compile(r"\[Core\]\s+DMA\s+(\d)\s+Instruction Retired")

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

    # Event Sync blocked
    event_sync_block_re = re.compile(
        r"\[(SAInstQueue|SFUInstQueue)\]\s+Inst Issue Blocked by Event Sync Unit"
    )

    def __init__(self) -> None:
        self.current_cycle = 0
        self.last_cycle = 0

        self.intervals: List[Interval] = []
        self.markers: List[Marker] = []
        self.predictions: List[Prediction] = []

        self.pending_dma_ops: Deque[str] = deque()

        self.open_dma: Dict[str, Tuple[int, str]] = {}
        self.open_sa: Optional[Tuple[int, str, int]] = None
        self.open_acc: Optional[Tuple[int, str, int]] = None
        self.open_sfu: Optional[Tuple[int, str, int]] = None
        self.open_ddr: Optional[Tuple[int, str]] = None

        self.open_waits: Dict[str, int] = {}

    def _close_interval(
        self,
        lane: str,
        start: int,
        label: str,
        end: int,
        kind: str = "exec",
        note: str = "",
        color: Optional[str] = None,
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
            )
        )

    def _add_marker(self, lane: str, cycle: int, label: str, kind: str, color: str) -> None:
        self.markers.append(Marker(lane=lane, cycle=cycle, label=label, kind=kind, color=color))

    def _add_prediction(self, lane: str, start: int, remaining: int, label: str, color: str) -> None:
        pred_end = start + remaining
        self.predictions.append(
            Prediction(
                lane=lane,
                start=start,
                predicted_end=pred_end,
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
                color=pick_color("WAIT"),
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

                # -------- Event Sync blocked --------
                m = self.event_sync_block_re.search(line)
                if m:
                    queue_name = m.group(1)
                    if queue_name == "SAInstQueue":
                        self._open_wait("SA")
                    elif queue_name == "SFUInstQueue":
                        self._open_wait("SFU")
                    continue

                # -------- DMA --------
                m = self.dma_op_re.search(line)
                if m:
                    self.pending_dma_ops.append(m.group(1))
                    continue

                m = self.dma_issue_re.search(line)
                if m:
                    dma_id = m.group(1)
                    lane = f"DMA{dma_id}"
                    label = self.pending_dma_ops.popleft() if self.pending_dma_ops else "DMA"
                    if lane in self.open_dma:
                        old_start, old_label = self.open_dma.pop(lane)
                        self._close_interval(lane, old_start, old_label, self.current_cycle)
                    self.open_dma[lane] = (self.current_cycle, label)
                    self._add_marker(lane, self.current_cycle, label, "issue", pick_color(label))
                    continue

                m = self.dma_retire_re.search(line)
                if m:
                    dma_id = m.group(1)
                    lane = f"DMA{dma_id}"
                    if lane in self.open_dma:
                        start, label = self.open_dma.pop(lane)
                        self._close_interval(lane, start, label, self.current_cycle + 1)
                        self._add_marker(lane, self.current_cycle, label, "retire", pick_color(label))
                    continue

                # -------- SA --------
                m = self.sa_start_re.search(line)
                if m:
                    label = m.group(1)
                    remaining = int(m.group(2))
                    self._close_wait("SA")
                    if self.open_sa is not None:
                        start, old_label, _ = self.open_sa
                        self._close_interval("SA", start, old_label, self.current_cycle)
                    self.open_sa = (self.current_cycle, label, remaining)
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
                        start, label, _ = self.open_sa
                        self._close_interval("SA", start, label, self.current_cycle + 1)
                        self._add_marker("SA", self.current_cycle, label, "finish", pick_color(label))
                        self.open_sa = None
                    continue

                # -------- ACC --------
                m = self.acc_start_re.search(line)
                if m:
                    label = m.group(1)
                    remaining = int(m.group(2))
                    if self.open_acc is not None:
                        start, old_label, _ = self.open_acc
                        self._close_interval("ACC", start, old_label, self.current_cycle)
                    self.open_acc = (self.current_cycle, label, remaining)
                    color = pick_color(label)
                    self._add_marker("ACC", self.current_cycle, label, "start", color)
                    self._add_prediction("ACC", self.current_cycle, remaining, label, color)
                    continue

                m = self.sa_retire_re.search(line)
                if m:
                    if self.open_acc is not None:
                        start, label, _ = self.open_acc
                        self._close_interval("ACC", start, label, self.current_cycle + 1)
                        self._add_marker("ACC", self.current_cycle, label, "retire", pick_color(label))
                        self.open_acc = None
                    continue

                # -------- SFU --------
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

                # -------- DDR --------
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

        for lane, (start, label) in list(self.open_dma.items()):
            self._close_interval(lane, start, label, end_cycle)

        if self.open_sa is not None:
            start, label, remaining = self.open_sa
            self._close_interval("SA", start, label, max(end_cycle, start + remaining))

        if self.open_acc is not None:
            start, label, remaining = self.open_acc
            self._close_interval("ACC", start, label, max(end_cycle, start + remaining))

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
                color=pick_color("WAIT"),
                kind="wait",
                note="Blocked by Event Sync Unit",
            )

        return self.intervals, self.markers, self.predictions, end_cycle


# ==========================================================
# Filter
# ==========================================================
def filter_intervals(intervals: List[Interval], start_cycle: int, end_cycle: int) -> List[Interval]:
    filtered: List[Interval] = []
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
                )
            )
    return filtered


def filter_markers(markers: List[Marker], start_cycle: int, end_cycle: int) -> List[Marker]:
    return [m for m in markers if start_cycle <= m.cycle <= end_cycle]


def filter_predictions(preds: List[Prediction], start_cycle: int, end_cycle: int) -> List[Prediction]:
    result = []
    for p in preds:
        if p.predicted_end < start_cycle:
            continue
        if p.start > end_cycle:
            continue
        result.append(p)
    return result


# ==========================================================
# Plotly Plot
# ==========================================================
def build_plotly_figure(
    intervals: List[Interval],
    markers: List[Marker],
    predictions: List[Prediction],
    title: str,
    lanes: List[str],
    show_markers: bool,
    show_predictions: bool,
    show_waits: bool,
    only_active_lanes: bool,
) -> go.Figure:
    lane_to_intervals: Dict[str, List[Interval]] = defaultdict(list)
    lane_to_markers: Dict[str, List[Marker]] = defaultdict(list)
    lane_to_predictions: Dict[str, List[Prediction]] = defaultdict(list)

    for iv in intervals:
        lane_to_intervals[iv.lane].append(iv)
    for mk in markers:
        lane_to_markers[mk.lane].append(mk)
    for pr in predictions:
        lane_to_predictions[pr.lane].append(pr)

    if only_active_lanes:
        active_lanes = [
            lane for lane in lanes
            if lane_to_intervals.get(lane) or lane_to_markers.get(lane) or lane_to_predictions.get(lane)
        ]
    else:
        active_lanes = lanes[:]

    if not active_lanes:
        raise RuntimeError("No timeline events found in the selected range.")

    xmin = min(iv.start for iv in intervals) if intervals else 0
    xmax = max(iv.end for iv in intervals) if intervals else xmin + 1

    y_pos = {lane: len(active_lanes) - 1 - idx for idx, lane in enumerate(active_lanes)}
    bar_h = 0.58

    fig = go.Figure()

    # baseline
    for lane in active_lanes:
        y = y_pos[lane]
        fig.add_shape(
            type="line",
            x0=xmin,
            x1=xmax,
            y0=y,
            y1=y,
            line=dict(color="#D9D9D9", width=1),
            layer="below",
        )

    # group intervals by (kind, label, color)
    grouped: Dict[Tuple[str, str, str], List[Interval]] = defaultdict(list)
    for iv in intervals:
        if iv.kind == "wait" and not show_waits:
            continue
        grouped[(iv.kind, iv.label, iv.color)].append(iv)

    for (kind, label, color), items in grouped.items():
        x = [iv.end - iv.start for iv in items]
        base = [iv.start for iv in items]
        y = [y_pos[iv.lane] for iv in items]
        text = []
        for iv in items:
            note = f"<br>{iv.note}" if iv.note else ""
            text.append(
                f"lane={iv.lane}<br>"
                f"label={iv.label}<br>"
                f"start={iv.start}<br>"
                f"end={iv.end}<br>"
                f"duration={iv.end - iv.start}{note}"
            )

        marker = dict(
            color=color,
            line=dict(color=color, width=1.2),
        )

        opacity = 0.82
        name = label

        if kind == "wait":
            marker["pattern"] = dict(shape="/", fgcolor="#888888", solidity=0.18)
            marker["line"] = dict(color="#999999", width=1.0)
            opacity = 0.40
            name = "EventSyncWait"

        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                base=base,
                orientation="h",
                width=bar_h,
                marker=marker,
                opacity=opacity,
                name=name,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=text,
                showlegend=True if label == name else True,
            )
        )

    # markers
    if show_markers:
        marker_symbol = {
            "start": "triangle-up",
            "finish": "triangle-down",
            "issue": "circle",
            "retire": "square",
        }

        grouped_markers: Dict[Tuple[str, str], List[Marker]] = defaultdict(list)
        for mk in markers:
            grouped_markers[(mk.kind, mk.color)].append(mk)

        for (kind, color), items in grouped_markers.items():
            fig.add_trace(
                go.Scatter(
                    x=[m.cycle for m in items],
                    y=[y_pos[m.lane] for m in items],
                    mode="markers",
                    marker=dict(
                        symbol=marker_symbol.get(kind, "circle"),
                        size=9,
                        color=color,
                        line=dict(color="black", width=0.8),
                    ),
                    name=f"marker:{kind}",
                    hovertemplate=(
                        "lane=%{text}<br>"
                        f"kind={kind}<br>"
                        "cycle=%{x}<extra></extra>"
                    ),
                    text=[f"{m.lane} | {m.label}" for m in items],
                    showlegend=False,
                )
            )

    # predictions
    if show_predictions:
        pred_x = []
        pred_y = []
        pred_text = []

        for pr in predictions:
            if pr.lane not in y_pos:
                continue
            y = y_pos[pr.lane]

            # vertical dashed line
            fig.add_shape(
                type="line",
                x0=pr.predicted_end,
                x1=pr.predicted_end,
                y0=y - bar_h / 2.0 - 0.12,
                y1=y + bar_h / 2.0 + 0.12,
                line=dict(color=pr.color, width=1.3, dash="dash"),
                opacity=0.95,
            )

            # top dashed guide
            fig.add_shape(
                type="line",
                x0=pr.start,
                x1=pr.predicted_end,
                y0=y + bar_h / 2.0 + 0.08,
                y1=y + bar_h / 2.0 + 0.08,
                line=dict(color=pr.color, width=1.0, dash="dot"),
                opacity=0.85,
            )

            pred_x.append(pr.predicted_end)
            pred_y.append(y + bar_h / 2.0 + 0.08)
            pred_text.append(
                f"lane={pr.lane}<br>"
                f"label={pr.label}<br>"
                f"predicted_end={pr.predicted_end}<br>"
                f"remaining={pr.predicted_end - pr.start}"
            )

        if pred_x:
            fig.add_trace(
                go.Scatter(
                    x=pred_x,
                    y=pred_y,
                    mode="markers",
                    marker=dict(
                        symbol="line-ns-open",
                        size=18,
                        color="#333333",
                        line=dict(color="#333333", width=1.0),
                    ),
                    name="predicted_end",
                    hovertemplate="%{text}<extra></extra>",
                    text=pred_text,
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        barmode="overlay",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        xaxis=dict(
            title="Cycle",
            showgrid=True,
            gridcolor="#EAEAEA",
            zeroline=False,
        ),
        yaxis=dict(
            title="Module",
            tickmode="array",
            tickvals=[y_pos[l] for l in active_lanes],
            ticktext=active_lanes,
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
        margin=dict(l=80, r=40, t=70, b=110),
        height=max(420, 120 * len(active_lanes)),
        hovermode="closest",
    )

    return fig


# ==========================================================
# Main
# ==========================================================
def main() -> None:
    log_path = Path(CONFIG["log"])
    out_html = Path(CONFIG["out_html"])
    out_png = Path(CONFIG["out_png"]) if CONFIG["out_png"] else None

    lanes = list(CONFIG["lanes"])
    start_cycle = int(CONFIG["start_cycle"])
    end_cycle_cfg = CONFIG["end_cycle"]

    parser = TimelineParser()
    intervals, markers, predictions, last_cycle = parser.parse(log_path)

    end_cycle = last_cycle if end_cycle_cfg is None else int(end_cycle_cfg)
    if end_cycle <= start_cycle:
        raise ValueError("CONFIG['end_cycle'] must be greater than CONFIG['start_cycle'].")

    intervals = filter_intervals(intervals, start_cycle, end_cycle)
    markers = filter_markers(markers, start_cycle, end_cycle)
    predictions = filter_predictions(predictions, start_cycle, end_cycle)

    title = CONFIG["title"] or f"{log_path.name} [{start_cycle}, {end_cycle})"

    fig = build_plotly_figure(
        intervals=intervals,
        markers=markers,
        predictions=predictions,
        title=title,
        lanes=lanes,
        show_markers=bool(CONFIG["show_markers"]),
        show_predictions=bool(CONFIG["show_predictions"]),
        show_waits=bool(CONFIG["show_waits"]),
        only_active_lanes=bool(CONFIG["only_active_lanes"]),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs=CONFIG["html_include_plotlyjs"])
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
