from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "frozen" / "aggregate_results"
OUTPUT = ROOT / "build" / "figures"


def read(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    pilot = read("cascade_pilot_adjudication.json")
    fresh = read("frozen_fresh_adjudication.json")
    windows = ["1", "2", "4", "8", "64"]
    x = np.arange(len(windows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for offset, label, data, color in (
        (-width / 2, "Cascade Pilot", pilot, "#3159a7"),
        (width / 2, "Frozen Fresh", fresh, "#d2642a"),
    ):
        values = [100 * data["window_metrics"]["normalized_total_cost"][m]["median_reduction"] for m in windows]
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, [f"m={value}" for value in windows])
    ax.set_ylabel("Median normalized-cost reduction (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "cost_reduction.png"
    fig.savefig(target, dpi=220)
    print(target)


if __name__ == "__main__":
    main()
