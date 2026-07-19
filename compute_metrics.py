"""
compute_metrics.py -- Computes accuracy and F1-score from evaluate.py
results files, matching the metric style used in the MDTeamGPT paper
(Table 1: Accuracy % and F1-score % per dataset).

USAGE
-----
    python compute_metrics.py --results results/medqa_baseline.jsonl
    python compute_metrics.py --results results/medqa_baseline.jsonl results/medqa_full.jsonl

Compares multiple files side by side if more than one is given.
"""

import argparse
import json
from sklearn.metrics import f1_score, accuracy_score


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def compute(path):
    records = load_jsonl(path)
    valid = [r for r in records if "error" not in r and r.get("final_answer")]

    y_true = [r["gold_answer"].strip().upper() for r in valid]
    y_pred = [r["final_answer"].strip().upper() for r in valid]

    acc = accuracy_score(y_true, y_pred) * 100
    # weighted F1 matches how the original paper reports F1 across
    # imbalanced multi-class answer distributions (A-E, or yes/no/maybe)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100

    n_total = len(records)
    n_errors = n_total - len(valid)
    n_conflict = sum(1 for r in valid if r.get("had_conflict_any_round"))
    n_da_fired = sum(1 for r in valid if r.get("devil_advocate_transcript"))
    n_tool_fired = sum(1 for r in valid if r.get("tool_query_transcript"))

    return {
        "path": path,
        "n_total": n_total,
        "n_graded": len(valid),
        "n_errors": n_errors,
        "accuracy": acc,
        "f1_weighted": f1,
        "n_conflict_detected": n_conflict,
        "n_devils_advocate_fired": n_da_fired,
        "n_conflict_tools_fired": n_tool_fired,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True,
                    help="One or more results JSONL files to compute metrics for.")
    args = p.parse_args()

    all_stats = [compute(path) for path in args.results]

    print(f"{'File':<45} {'N':>5} {'Graded':>7} {'Errors':>7} "
          f"{'Acc%':>7} {'F1%':>7} {'Conflict':>9} {'DA':>4} {'Tools':>6}")
    print("-" * 110)
    for s in all_stats:
        print(f"{s['path']:<45} {s['n_total']:>5} {s['n_graded']:>7} "
              f"{s['n_errors']:>7} {s['accuracy']:>7.1f} {s['f1_weighted']:>7.1f} "
              f"{s['n_conflict_detected']:>9} {s['n_devils_advocate_fired']:>4} "
              f"{s['n_conflict_tools_fired']:>6}")

    if len(all_stats) == 2:
        print()
        print("=== DELTA (second file - first file) ===")
        acc_delta = all_stats[1]["accuracy"] - all_stats[0]["accuracy"]
        f1_delta = all_stats[1]["f1_weighted"] - all_stats[0]["f1_weighted"]
        print(f"Accuracy delta: {acc_delta:+.1f} percentage points")
        print(f"F1 delta:       {f1_delta:+.1f} percentage points")


if __name__ == "__main__":
    main()
