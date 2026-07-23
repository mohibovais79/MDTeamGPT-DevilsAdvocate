"""
regrade_results.py -- Re-grades an EXISTING evaluate.py results file,
fixing TWO known grading bugs, without re-running the expensive
multi-agent consultation pipeline:

BUG 1 (fixed earlier): cot_reviewer truncated case_info to 500 chars,
cutting off MCQ options before the grading judge could see them.

BUG 2 (this fix): when final_answer is blank or a non-answer (e.g. the
Safety Reviewer failed to fill in a clean letter, or the system produced
"Not applicable" / "This case does not have multiple-choice options." /
the max-rounds fallback message), the old code still asked an LLM judge
"is this correct?" -- and the judge sometimes unreliably said yes. A
blank or non-answer can never be a correct answer; this version auto-
fails these cases deterministically, WITHOUT invoking the judge at all.

USAGE
-----
    python regrade_results.py \\
        --results results/mistral_7b_100_baseline_20260721_031457.jsonl \\
        --dataset data/medqa_test.jsonl \\
        --output results/mistral_7b_100_baseline_regraded.jsonl \\
        --provider ollama --model mistral:7b \\
        --base_url http://localhost:11434
"""

import argparse
import json
import re

from agents import MDTAgents


_LETTER_RE = re.compile(r"\b([A-E])\b")

# Known non-answer patterns. A blank string, or any of these (after
# stripping whitespace/trailing period and lowercasing), is auto-graded
# as incorrect -- it is never sent to the LLM judge, because there is
# nothing for the judge to correctly evaluate.
_NON_ANSWER_PATTERNS = {
    "", "not applicable", "n/a", "na",
    "this case does not have multiple-choice options",
    "max rounds reached. proceeding with latest hypothesis",
    "max rounds reached, proceeding with latest hypothesis",
}


def _is_non_answer(text: str) -> bool:
    if not text or not text.strip():
        return True
    normalized = text.strip().lower().rstrip(".")
    return normalized in _NON_ANSWER_PATTERNS


def _extract_choice_letter(text):
    if not text:
        return None
    match = _LETTER_RE.search(text.strip().upper())
    return match.group(1) if match else None


def grade_mcq_deterministic(final_answer, gold_answer):
    pred = _extract_choice_letter(final_answer)
    if pred is None:
        return None
    gold = _extract_choice_letter(gold_answer) or gold_answer.strip().upper()
    return pred == gold


def format_case_info(case):
    text = case["case_info"].strip()
    options = case.get("options")
    if options:
        opt_lines = "\n".join(f"{k}: {v}" for k, v in options.items())
        text = f"{text}\n\nOptions:\n{opt_lines}"
    return text


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Existing evaluate.py output JSONL.")
    p.add_argument("--dataset", required=True, help="The original dataset JSONL (for full options).")
    p.add_argument("--output", required=True, help="Where to write the re-graded JSONL.")
    p.add_argument("--provider", default="openai", choices=["openai", "ollama"])
    p.add_argument("--model", required=True)
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", default=None, help="Defaults to 'ollama' if provider=ollama.")
    args = p.parse_args()

    api_key = args.api_key or ("ollama" if args.provider == "ollama" else None)
    if args.provider == "openai" and not api_key:
        raise SystemExit("Pass --api_key for provider=openai (or set it via config.json).")

    results = load_jsonl(args.results)
    dataset = {c["id"]: c for c in load_jsonl(args.dataset)}

    agents = MDTAgents(
        api_key=api_key,
        base_url=args.base_url,
        text_model=args.model,
        vl_model=args.model,
        enable_tools=False,
        provider=args.provider,
    )

    n_correct_old = 0
    n_correct_new = 0
    n_total = 0
    changed = 0
    n_auto_failed = 0

    print(f"Re-grading {len(results)} records from {args.results} ...")

    with open(args.output, "w", encoding="utf-8") as out_f:
        for r in results:
            if "error" in r:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue

            case = dataset.get(r["id"])
            if case is None:
                print(f"WARNING: {r['id']} not found in {args.dataset}, skipping regrade.")
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue

            n_total += 1
            old_correct = r.get("correct")
            if old_correct:
                n_correct_old += 1

            final_answer = r.get("final_answer", "")
            gold_answer = r.get("gold_answer", case.get("gold_answer", ""))

            # --- BUG 2 FIX: auto-fail blank/non-answers, never ask the judge ---
            if _is_non_answer(final_answer):
                new_correct = False
                n_auto_failed += 1
            else:
                new_correct = grade_mcq_deterministic(final_answer, gold_answer)
                if new_correct is None:
                    case_info_full = format_case_info(case)
                    judge = agents.cot_reviewer(
                        case_info=case_info_full,
                        final_answer=final_answer,
                        ground_truth=gold_answer,
                    )
                    new_correct = bool(judge.get("is_correct", False))

            if new_correct:
                n_correct_new += 1
            if new_correct != old_correct:
                changed += 1
                print(f"  {r['id']}: {old_correct} -> {new_correct}  "
                      f"(answer: {final_answer[:70]!r})")

            r["correct"] = new_correct
            r["correct_before_regrade"] = old_correct
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"Total re-graded: {n_total}")
    print(f"Auto-failed (blank/non-answer, judge never consulted): {n_auto_failed}")
    if n_total:
        print(f"Old accuracy: {n_correct_old}/{n_total} = {n_correct_old / n_total * 100:.1f}%")
        print(f"New accuracy: {n_correct_new}/{n_total} = {n_correct_new / n_total * 100:.1f}%")
    print(f"Cases where the grade changed: {changed}")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()