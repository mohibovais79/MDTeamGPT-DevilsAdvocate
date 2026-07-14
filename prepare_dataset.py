"""
prepare_dataset.py — Converts MedQA / PubMedQA (via the HuggingFace
`datasets` library) into the JSONL schema evaluate.py expects.
 
Requires: pip install datasets
 
USAGE
-----
    python prepare_dataset.py --source medqa     --split test --limit 150 --out data/medqa_test.jsonl
    python prepare_dataset.py --source pubmedqa  --split train --limit 150 --out data/pubmedqa_test.jsonl
 
NOTE: There are several community mirrors of both datasets on the
HuggingFace Hub with slightly different config names / column names. The
identifiers below are commonly used mirrors as of early 2026; if either
`load_dataset(...)` call fails, search https://huggingface.co/datasets for
"MedQA" or "PubMedQA" and swap in the mirror + config name that works,
then adjust the column names in the row-mapping loop below to match.
"""
 
import argparse
import json
 
from datasets import load_dataset
 
 
def prepare_medqa(split, limit):
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    cases = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        options = row["options"]  # e.g. {"A": "...", "B": "...", ...}
        cases.append({
            "id": f"medqa_{i}",
            "case_info": row["question"],
            "question_type": "mcq",
            "options": options,
            "gold_answer": row["answer_idx"],
        })
    return cases
 
 
def prepare_pubmedqa(split, limit):
    # qiaojin/PubMedQA's pqa_labeled subset only has a 'train' split.
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    cases = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        context = " ".join(row["context"]["contexts"])
        cases.append({
            "id": f"pubmedqa_{i}",
            "case_info": f"{row['question']}\n\nAbstract Context:\n{context}",
            "question_type": "mcq",
            "options": {"yes": "Yes", "no": "No", "maybe": "Maybe"},
            "gold_answer": row["final_decision"],
        })
    return cases
 
 
SOURCES = {
    "medqa": prepare_medqa,
    "pubmedqa": prepare_pubmedqa,
}
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=list(SOURCES.keys()), required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
 
    cases = SOURCES[args.source](args.split, args.limit)
 
    with open(args.out, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
 
    print(f"Wrote {len(cases)} cases to {args.out}")
    if cases:
        print("First record (sanity check):")
        print(json.dumps(cases[0], ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()
 
































