"""
compute_metrics.py -- Computes accuracy and F1-score from evaluate.py
results files, matching the metric style used in the MDTeamGPT paper
(Table 1: Accuracy % and F1-score % per dataset).

FIXED (this version): the "valid" filter previously excluded any record
where final_answer was falsy -- which in Python means an EMPTY STRING ""
was silently treated the same as a real pipeline crash, shrinking the
denominator and inflating the reported accuracy percentage. A blank
answer is not an error: it is a real, gradeable outcome (and, after the
regrade_results.py fix, a correctly-graded "incorrect" outcome). This
version only excludes records with a literal "error" key (an actual
exception during the run) from the denominator; blank/non-answers are
counted as graded, using the trusted "correct" field.

USAGE
-----
    python compute_metrics.py --results results/medqa_baseline.jsonl
    python compute_metrics.py --results results/medqa_baseline.jsonl results/medqa_full.jsonl

Compares multiple files side by side if more than one is given.
"""

import argparse
import json
from sklearn.metrics import f1_score


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

    # FIXED: only exclude records with a literal "error" key (a real
    # exception during the run). Blank/non-answer final_answer values are
    # still counted -- they have a well-defined "correct" field (False,
    # after the regrade fix) and should not silently shrink the
    # denominator.
    valid = [r for r in records if "error" not in r]

    n_correct = sum(1 for r in valid if r.get("correct"))
    acc = (n_correct / len(valid) * 100) if valid else 0

    y_true_labels = [r["gold_answer"].strip().upper() for r in valid]
    y_pred_labels = []
    for r in valid:
        pred = (r.get("final_answer") or "").strip().upper()
        gold = r["gold_answer"].strip().upper()
        if r.get("correct") and pred != gold:
            # Case was marked correct (per the trusted grading logic) but
            # the raw text doesn't exactly match gold (e.g. "A: Diltiazem"
            # instead of "A") -- substitute gold as the predicted label so
            # F1 reflects the trusted correctness judgment.
            y_pred_labels.append(gold)
        else:
            # Genuinely wrong or blank -- use the raw text as-is (or an
            # empty-string placeholder for blanks). Weighted F1 only
            # weights by support in y_true, so an unusual/blank predicted
            # label here does not distort the reported score.
            y_pred_labels.append(pred if pred else "(blank)")

    f1 = f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0) * 100

    n_total = len(records)
    n_errors = n_total - len(valid)
    n_blank_or_nonanswer = sum(
        1 for r in valid if not (r.get("final_answer") or "").strip()
    )
    n_conflict = sum(1 for r in valid if r.get("had_conflict_any_round"))
    n_da_fired = sum(1 for r in valid if r.get("devil_advocate_transcript"))
    n_tool_fired = sum(1 for r in valid if r.get("tool_query_transcript"))

    return {
        "path": path,
        "n_total": n_total,
        "n_graded": len(valid),
        "n_errors": n_errors,
        "n_blank_or_nonanswer": n_blank_or_nonanswer,
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

    print(f"{'File':<45} {'N':>5} {'Graded':>7} {'RealErr':>7} {'Blank':>6} "
          f"{'Acc%':>7} {'F1%':>7} {'Conflict':>9} {'DA':>4} {'Tools':>6}")
    print("-" * 120)
    for s in all_stats:
        print(f"{s['path']:<45} {s['n_total']:>5} {s['n_graded']:>7} "
              f"{s['n_errors']:>7} {s['n_blank_or_nonanswer']:>6} "
              f"{s['accuracy']:>7.1f} {s['f1_weighted']:>7.1f} "
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
