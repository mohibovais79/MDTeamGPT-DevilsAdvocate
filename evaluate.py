"""
evaluate.py — Headless batch evaluation harness for MDTeamGPT experiments.

This is the non-GUI counterpart to app.py. It runs a JSONL dataset of cases
through the LangGraph workflow under a selected experimental condition
("mode"), and writes one JSON record per case to an output JSONL file --
final answer, correctness, per-round conflict flags, Devil's Advocate
transcripts, and tool-query transcripts. Nothing here touches Streamlit.

MODES
-----
    baseline         original MDTeamGPT behavior: legacy per-specialist,
                     every-round tool calls active; no Devil's Advocate.
    tools_off        all external tool use disabled (isolates whether
                     undirected tool use helps or hurts at all).
    conflict_tools   legacy per-specialist tool calls disabled; a single
                     targeted tool query fires only when the Lead Physician
                     detects a conflict (Contribution 2).
    devils_advocate  Devil's Advocate agent active on detected conflict;
                     legacy tool-call behavior unchanged (Contribution 1).
    full             both conflict_tools and devils_advocate active.

PROVIDER
--------
    --provider openai   (default) any OpenAI-compatible API: OpenAI, Groq,
                         DashScope, etc.
    --provider ollama   locally-served Ollama model. Required to properly
                         disable "thinking" mode on reasoning models (e.g.
                         Qwen3.5) -- ChatOpenAI's OpenAI-compat path does
                         NOT reliably support this; ChatOllama does.
                         Requires: uv add langchain-ollama

USAGE
-----
    python evaluate.py \\
        --dataset data/medqa_test.jsonl \\
        --mode baseline \\
        --model gpt-4o-mini \\
        --output results/medqa_baseline_gpt4omini.jsonl \\
        --limit 150

For a local open-weight model via Ollama:
    MDT_API_KEY=ollama python evaluate.py \\
        --dataset data/medqa_test.jsonl --mode full \\
        --model qwen3.5:9b-mlx --provider ollama \\
        --base_url http://localhost:11434 \\
        --output results/medqa_full_qwen9b.jsonl

DATASET SCHEMA (one JSON object per line)
------------------------------------------
    {
        "id": "medqa_0001",
        "case_info": "A 36-year-old man presents with ...",
        "question_type": "mcq",
        "options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
        "gold_answer": "B"
    }

See prepare_dataset.py for a script that builds this from MedQA / PubMedQA.
"""

import argparse
import json
import os
import re
import time
import traceback

from agents import MDTAgents
from workflow import create_workflow
from utils import load_config
from trace_logger import init_trace_logger, get_trace_logger


MODE_PRESETS = {
    "baseline":        dict(enable_tools=True,  enable_conflict_tools=False, enable_devils_advocate=False),
    "tools_off":       dict(enable_tools=False, enable_conflict_tools=False, enable_devils_advocate=False),
    "conflict_tools":  dict(enable_tools=True,  enable_conflict_tools=True,  enable_devils_advocate=False),
    "devils_advocate": dict(enable_tools=True,  enable_conflict_tools=False, enable_devils_advocate=True),
    "full":            dict(enable_tools=True,  enable_conflict_tools=True,  enable_devils_advocate=True),
}


# --------------------------------------------------------------------------
# Dataset loading / formatting
# --------------------------------------------------------------------------

def load_dataset(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def format_case_info(case: dict) -> str:
    """Builds the full case text the workflow sees: question plus, if
    present, the MCQ options formatted as a plain option list."""
    text = case["case_info"].strip()
    options = case.get("options")
    if options:
        opt_lines = "\n".join(f"{k}: {v}" for k, v in options.items())
        text = f"{text}\n\nOptions:\n{opt_lines}"
    return text


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

_LETTER_RE = re.compile(r"\b([A-E])\b")


def _extract_choice_letter(text: str):
    if not text:
        return None
    match = _LETTER_RE.search(text.strip().upper())
    return match.group(1) if match else None


def grade_mcq_deterministic(final_answer: str, gold_answer: str):
    """
    Best-effort deterministic MCQ grading by extracting a single A-E letter
    from both strings and comparing. Returns True / False / None.
    None means "could not confidently extract a letter" -- the caller
    should fall back to the LLM-based CoT reviewer in that case (this is
    also what naturally handles PubMedQA's yes/no/maybe answers, since
    those never match the A-E pattern and always fall through to the LLM
    judge, mirroring how the original paper grades open-ended questions).
    """
    pred = _extract_choice_letter(final_answer)
    if pred is None:
        return None
    gold = _extract_choice_letter(gold_answer) or gold_answer.strip().upper()
    return pred == gold


# --------------------------------------------------------------------------
# Single-case execution
# --------------------------------------------------------------------------

def run_single_case(agents_instance, app, case: dict, use_knowledge_base: bool,
                     max_rounds: int, image_base64=None) -> dict:
    case_info_full = format_case_info(case)

    state = {
        "case_info": case_info_full,
        "image_base64": image_base64,
        "ground_truth": case.get("gold_answer", ""),
        "selected_roles": [],
        "triage_reason": "",
        "current_round": 1,
        "max_rounds": max_rounds,
        "context_bullets": [],
        "final_answer": "",
        "is_converged": False,
        "kb_context_text": "",
        "kb_context_docs": [],
        "use_knowledge_base": use_knowledge_base,
        "pending_evidence": "",
        "current_devil_advocate_text": "",
        "round_conflict_flags": [],
        "devil_advocate_transcript": [],
        "tool_query_transcript": [],
    }

    # Generous recursion-limit margin: each round involves 3 node
    # executions (consultation -> conflict_response -> safety), so
    # max_rounds rounds need roughly max_rounds * 3 steps plus triage.
    final_state = app.invoke(state, config={"recursion_limit": (max_rounds + 2) * 6})
    return final_state


# --------------------------------------------------------------------------
# Main evaluation loop
# --------------------------------------------------------------------------

def evaluate(args):
    cfg = load_config()
    api_key = os.environ.get("MDT_API_KEY", cfg.get("api_key", ""))
    base_url = args.base_url or cfg.get("base_url")
    text_model = args.model or cfg.get("text_model")
    vl_model = args.vl_model or cfg.get("vl_model", text_model)

    if args.provider == "openai" and not api_key:
        raise SystemExit(
            "No API key found. Set the MDT_API_KEY environment variable, "
            "or api_key in config.json (copy config.example.json first). "
            "If you're running a local Ollama model, pass --provider ollama."
        )

    preset = MODE_PRESETS[args.mode]

    agents_instance = MDTAgents(
        api_key=api_key,
        base_url=base_url,
        text_model=text_model,
        vl_model=vl_model,
        enable_tools=preset["enable_tools"],
        enable_conflict_tools=preset["enable_conflict_tools"],
        enable_devils_advocate=preset["enable_devils_advocate"],
        provider=args.provider,
    )
    app = create_workflow(agents_instance)

    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    num_runs = args.limit if args.limit else len(cases)
    log_path = init_trace_logger(text_model, num_runs, args.mode)
    print(f"Trace log: {log_path}")

    if args.resume:
        output_path = args.resume
        completed_ids = set()
        existing_records = []
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as prev_f:
                for line in prev_f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        existing_records.append(rec)
                        completed_ids.add(rec.get("id"))
        print(f"Resuming: {len(completed_ids)} cases already done in {output_path}")
    elif args.output:
        output_path = args.output
        completed_ids = set()
        existing_records = []
    else:
        from datetime import datetime
        os.makedirs("results", exist_ok=True)
        safe_model = text_model.replace("/", "_").replace(":", "_").replace(".", "_")
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/{safe_model}_{num_runs}_{args.mode}_{date_str}.jsonl"
        completed_ids = set()
        existing_records = []

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_correct = sum(1 for r in existing_records if r.get("correct"))
    n_graded = sum(1 for r in existing_records if "error" not in r and r.get("final_answer"))

    remaining = [(i, c) for i, c in enumerate(cases) if c.get("id", i) not in completed_ids]

    print(f"Running {len(remaining)}/{len(cases)} cases | mode={args.mode} | provider={args.provider} "
          f"| model={text_model} | use_kb={args.use_kb} | max_rounds={args.max_rounds}")

    print(f"Results file: {output_path}")

    with open(output_path, "a", encoding="utf-8") as out_f:
        for idx, (i, case) in enumerate(remaining):
            case_id = case.get("id", i)
            print(f"[{len(completed_ids) + idx + 1}/{len(cases)}] case={case_id} ...", end=" ", flush=True)
            t0 = time.time()

            try:
                final_state = run_single_case(
                    agents_instance, app, case,
                    use_knowledge_base=args.use_kb,
                    max_rounds=args.max_rounds,
                )

                final_answer = final_state.get("final_answer", "")
                gold_answer = case.get("gold_answer", "")
                case_info_full = format_case_info(case)

                correct = None
                if case.get("question_type", "mcq") == "mcq":
                    correct = grade_mcq_deterministic(final_answer, gold_answer)

                if correct is None:
                    # Fall back to the framework's own CoT-reviewer LLM
                    # judge -- this is the same grading agent the original
                    # authors use, so it stays consistent with their
                    # methodology for open-ended / non-lettered answers.
                    judge_result = agents_instance.cot_reviewer(
                        case_info=case_info_full,
                        final_answer=final_answer,
                        ground_truth=gold_answer,
                    )
                    correct = bool(judge_result.get("is_correct", False))

                n_graded += 1
                if correct:
                    n_correct += 1

                tlog = get_trace_logger()
                tlog.info(f"=== CASE {case_id} | GRADE: {'CORRECT' if correct else 'WRONG'} | final={final_answer} | gold={gold_answer} | rounds={final_state.get('current_round')} ===")

                record = {
                    "id": case_id,
                    "mode": args.mode,
                    "provider": args.provider,
                    "model": text_model,
                    "final_answer": final_answer,
                    "gold_answer": gold_answer,
                    "correct": correct,
                    "rounds_used": final_state.get("current_round"),
                    "converged": final_state.get("is_converged"),
                    "had_conflict_any_round": any(final_state.get("round_conflict_flags", [])),
                    "round_conflict_flags": final_state.get("round_conflict_flags", []),
                    "devil_advocate_transcript": final_state.get("devil_advocate_transcript", []),
                    "tool_query_transcript": final_state.get("tool_query_transcript", []),
                    "context_bullets": final_state.get("context_bullets", []),
                    "elapsed_seconds": round(time.time() - t0, 2),
                }
                print(f"correct={correct} rounds={record['rounds_used']} "
                      f"({record['elapsed_seconds']}s)")

            except Exception as e:
                record = {
                    "id": case_id,
                    "mode": args.mode,
                    "provider": args.provider,
                    "model": text_model,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "elapsed_seconds": round(time.time() - t0, 2),
                }
                print(f"ERROR: {e}")

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    acc = (n_correct / n_graded * 100) if n_graded else 0.0
    print(f"\nDone. Graded {n_graded}/{len(cases)} cases. "
          f"Accuracy: {n_correct}/{n_graded} = {acc:.1f}%")
    print(f"Results written to {output_path}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Batch evaluation harness for MDTeamGPT.")
    p.add_argument("--dataset", required=True, help="Path to JSONL dataset file.")
    p.add_argument("--output", default=None, help="Path to write JSONL results. If omitted, auto-generated in results/ with model, mode, count, and timestamp.")
    p.add_argument("--mode", required=True, choices=list(MODE_PRESETS.keys()))
    p.add_argument("--provider", default="openai", choices=["openai", "ollama"],
                    help="'openai' for any OpenAI-compatible API. 'ollama' "
                         "for locally-served Ollama models -- required to "
                         "properly disable 'thinking' mode on reasoning "
                         "models like Qwen3.5.")
    p.add_argument("--model", default=None, help="Text model ID (overrides config.json).")
    p.add_argument("--vl_model", default=None, help="Vision model ID (overrides config.json).")
    p.add_argument("--base_url", default=None, help="API base URL (overrides config.json).")
    p.add_argument("--max_rounds", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="Only run the first N cases.")
    p.add_argument("--use_kb", action="store_true",
                    help="Enable CorrectKB/ChainKB read+write during this run. "
                         "Default OFF -- recommended OFF for the core ablation "
                         "so experience-accumulation effects don't confound "
                         "the comparison between conditions.")
    p.add_argument("--resume", default=None,
                    help="Path to an existing results JSONL file. Completed cases "
                         "are loaded and skipped; new cases are appended to the same file.")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    evaluate(cli_args)