"""Compare complete, full precision repeated-baseline and feature/resume reports."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base0", type=Path)
    p.add_argument("base1", type=Path)
    p.add_argument("feature", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reports = [
        json.loads((path / "rank0.json").read_text())
        for path in (args.base0, args.base1, args.feature)
    ]
    steps = reports[0]["steps"]
    for report in reports:
        assert report["result"] == "PASSED" and report["steps"] == steps and report["start"] == 0
        assert [row["train/step"] for row in report["rows"]] == list(range(steps)), "Incomplete run"
    ranks = sorted(args.base0.glob("rank*.json"))
    assert [p.name for p in ranks] == sorted(p.name for p in args.base1.glob("rank*.json"))
    assert [p.name for p in ranks] == sorted(p.name for p in args.feature.glob("rank*.json"))
    for rank in ranks:
        arms = [
            json.loads((path / rank.name).read_text())
            for path in (args.base0, args.base1, args.feature)
        ]
        for arm in arms:
            assert arm["result"] == "PASSED" and arm["steps"] == steps and arm["start"] == 0
        initial = [arm["initial_state"] for arm in arms]
        assert initial[0] == initial[1] == initial[2], f"Unmatched initial weights on {rank.name}"
        hooks = [arm["audited_view_hooks"] for arm in arms]
        assert hooks[0] == hooks[1] == hooks[2] and hooks[0] > 0, (
            f"Unmatched or missing decoder view audits on {rank.name}: {hooks}"
        )
        batches = [arm["batches"] for arm in arms]
        assert batches[0] == batches[1] == batches[2], f"Unmatched inputs on {rank.name}"
    metrics = {}
    passed = True
    for key in ("train/cross-entropy-loss", "train/load-balance-loss", "train/gradient-norm"):
        values = [np.array([row[key] for row in report["rows"]]) for report in reports]
        assert all(np.isfinite(value).all() for value in values)
        floor = float(np.abs(values[0][1:] - values[1][1:]).mean())
        delta = float(np.abs(values[0][1:] - values[2][1:]).mean())
        ratio = delta / floor if floor else (0.0 if delta == 0 else None)
        ok = ratio is not None and ratio < 3
        metrics[key] = dict(
            baseline_mean_abs_delta=floor,
            feature_mean_abs_delta=delta,
            ratio=ratio,
            passed=ok,
            first_step_delta=float(abs(values[0][0] - values[2][0])),
        )
        passed &= ok
    if args.resume:
        resumed = json.loads((args.resume / "rank0.json").read_text())
        start = resumed["start"]
        assert resumed["exact_restore"] and resumed["steps"] == steps
        assert [row["train/step"] for row in resumed["rows"]] == list(range(start, steps))
        for key in ("train/cross-entropy-loss", "train/load-balance-loss", "train/gradient-norm"):
            baseline = [
                np.array([row[key] for row in report["rows"]][start:]) for report in reports[:2]
            ]
            feature = np.array([row[key] for row in reports[2]["rows"]][start:])
            restored = np.array([row[key] for row in resumed["rows"]])
            floor = float(np.abs(baseline[0] - baseline[1]).mean())
            delta = float(np.abs(feature - restored).mean())
            ratio = delta / floor if floor else (0.0 if delta == 0 else None)
            ok = ratio is not None and ratio < 3
            metrics[f"resume/{key}"] = dict(
                baseline_mean_abs_delta=floor, resume_mean_abs_delta=delta, ratio=ratio, passed=ok
            )
            passed &= ok
    output = dict(result="PASSED" if passed else "INVESTIGATE", steps=steps, metrics=metrics)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    assert passed, (
        "Numerical deltas exceed the repeated-baseline envelope; diagnose before changing gates"
    )


if __name__ == "__main__":
    main()
