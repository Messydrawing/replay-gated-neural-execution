from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "frozen" / "aggregate_results"


def read(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    reference = read("reference_stability_closeout.json")
    rows = [
        ("Cascade Pilot", read("cascade_pilot_adjudication.json")),
        ("Frozen Fresh", read("frozen_fresh_adjudication.json")),
    ]
    print(
        "Reference Stability: "
        f"{reference['semantic_reference']['cell_count']}/767 replay; "
        f"{reference['rediscovery']['rediscovered_count']}/"
        f"{reference['rediscovery']['repeat_count']} difficult-cell repeats rediscovered"
    )
    print("experiment       coverage  m=1 median  m=4 median  energy P99  auth-bypass")
    for name, result in rows:
        metrics = result["window_metrics"]["normalized_total_cost"]
        print(
            f"{name:<16} "
            f"{result['cascade_certified_count']:>3}/{result['population_cells']:<3}   "
            f"{pct(metrics['1']['median_reduction']):>10}  "
            f"{pct(metrics['4']['median_reduction']):>10}  "
            f"{result['committed_energy_p99']:.9f}  "
            f"{len(result['unsafe_commits'])}"
        )


if __name__ == "__main__":
    main()
