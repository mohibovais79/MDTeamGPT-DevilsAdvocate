"""
mcnemar_test.py -- Paired significance testing between two evaluate.py
results files (e.g., baseline vs full), using McNemar's exact test.

This is the CORRECT test for this comparison -- not a t-test. Your baseline
and full runs are PAIRED (same case IDs, same model), and the outcome is
binary (correct/incorrect). McNemar's test looks only at the DISCORDANT
pairs (cases where the two conditions disagree) and asks whether the
direction of disagreement is asymmetric, which is exactly the question
"did our intervention help more than it hurt?"

USAGE
-----
    python mcnemar_test.py --baseline results/medqa_baseline.jsonl --full results/medqa_full.jsonl

    # Also print full detail for every case that flipped, to help pull
    # qualitative case studies for the paper:
    python mcnemar_test.py --baseline results/medqa_baseline.jsonl --full results/medqa_full.jsonl --show_cases
"""

import argparse
import json

from scipy.stats import binomtest


def load_jsonl(path):
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            items[r["id"]] = r
    return items


def print_case_detail(cid, base_record, full_record, label):
    print(f"\n--- {label}: {cid} ---")
    print(f"Gold answer:      {base_record.get('gold_answer')}")
    print(f"Baseline answer:  {base_record.get('final_answer')}  "
          f"(correct={base_record.get('correct')})")
    print(f"Full answer:      {full_record.get('final_answer')}  "
          f"(correct={full_record.get('correct')})")
    print(f"Baseline conflict detected: {base_record.get('had_conflict_any_round')}")
    print(f"Full conflict detected:     {full_record.get('had_conflict_any_round')}")
    da = full_record.get("devil_advocate_transcript", [])
    if da:
        print(f"Devil's Advocate fired ({len(da)} time(s)):")
        for entry in da:
            print(f"  {entry[:400]}")
    tq = full_record.get("tool_query_transcript", [])
    if tq:
        print(f"Conflict-directed tool query fired ({len(tq)} time(s)):")
        for entry in tq:
            print(f"  {entry[:300]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--full", required=True)
    p.add_argument("--show_cases", action="store_true",
                    help="Print full detail for every case that flipped between conditions.")
    args = p.parse_args()

    base = load_jsonl(args.baseline)
    full = load_jsonl(args.full)

    common_ids = sorted(set(base.keys()) & set(full.keys()))
    if len(common_ids) < len(base) or len(common_ids) < len(full):
        print(f"WARNING: only {len(common_ids)} case IDs are common to both files "
              f"(baseline has {len(base)}, full has {len(full)}). "
              f"Only common IDs are used for the paired test.\n")

    both_correct = 0
    both_wrong = 0
    base_right_full_wrong = 0   # "regression"
    base_wrong_full_right = 0   # "improvement"

    regressions = []
    improvements = []

    for cid in common_ids:
        b = base[cid]
        f = full[cid]
        if "error" in b or "error" in f:
            continue
        bc = bool(b.get("correct"))
        fc = bool(f.get("correct"))

        if bc and fc:
            both_correct += 1
        elif not bc and not fc:
            both_wrong += 1
        elif bc and not fc:
            base_right_full_wrong += 1
            regressions.append(cid)
        else:
            base_wrong_full_right += 1
            improvements.append(cid)

    n_discordant = base_right_full_wrong + base_wrong_full_right

    print(f"Paired cases analyzed: {len(common_ids)}")
    print(f"Both correct:                                 {both_correct}")
    print(f"Both wrong:                                    {both_wrong}")
    print(f"Baseline right -> Full wrong (regression):      {base_right_full_wrong}")
    print(f"Baseline wrong -> Full right (improvement):     {base_wrong_full_right}")
    print(f"Discordant pairs (used for the test):           {n_discordant}")
    print()

    if n_discordant == 0:
        print("No discordant pairs -- baseline and full agree on every case. "
              "McNemar's test is undefined (nothing to test).")
        return

    # McNemar's exact test: under H0, each discordant pair is equally
    # likely to go either direction (p=0.5 binomial). Two-sided exact test
    # on the smaller of the two discordant counts.
    k = min(base_right_full_wrong, base_wrong_full_right)
    result = binomtest(k, n_discordant, p=0.5, alternative="two-sided")

    print(f"McNemar's exact test (two-sided): p = {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("=> Statistically significant difference between conditions (p < 0.05).")
    else:
        print("=> NOT statistically significant at p < 0.05 with this sample size. "
              "Report the raw numbers honestly; do not claim significance you "
              "have not demonstrated. A larger N may be needed to detect a "
              "real but small effect.")

    print()
    print(f"Case IDs that IMPROVED (baseline wrong -> full right): {improvements}")
    print(f"Case IDs that REGRESSED (baseline right -> full wrong): {regressions}")

    if args.show_cases:
        for cid in improvements:
            print_case_detail(cid, base[cid], full[cid], "IMPROVEMENT")
        for cid in regressions:
            print_case_detail(cid, base[cid], full[cid], "REGRESSION")


if __name__ == "__main__":
    main()