# Experiments Guide — Devil's Advocate & Conflict-Directed Tool Retrieval
 
This is the mechanical, copy-paste checklist for running the experiments.
You don't need to hold the whole research plan in your head — just work
through this file top to bottom.
 
## What changed in the code (one paragraph)
 
`agents.py` gained two new methods (`generate_conflict_query`,
`devil_advocate_argument`) and one gated behavior change (the old blind
per-specialist tool call is now suppressed when conflict-directed
retrieval is active). `workflow.py` gained one new node,
`conflict_response_layer`, inserted between the consultation round and the
safety check — it detects whether the Lead Physician's `Conflict` field is
non-empty, and if so, fires whichever of the two contributions are
enabled. Both contributions default to **off**, so constructing the system
with no extra arguments reproduces the original, unmodified behavior
exactly. `evaluate.py` is the new headless script that runs a batch of
cases through any of five modes and logs everything to JSONL.
 
## 0. One-time setup
 
```bash
pip install -r requirements.txt   # or: uv sync
cp config.example.json config.json
# edit config.json and set YOUR OWN api_key locally — never commit this file
pip install datasets               # only needed for prepare_dataset.py
```
 
⚠️ **`config.json` must never be committed.** It's already in `.gitignore`.
If a real API key was ever committed to this repo's history, rotate /
revoke it on the provider's dashboard immediately — removing it in a later
commit does **not** remove it from git history.
 
You can also skip config.json entirely and just set an environment
variable before running anything:
```bash
export MDT_API_KEY="sk-..."
```
 
## 1. Prepare datasets (once)
 
```bash
python prepare_dataset.py --source medqa    --split test --limit 150 --out data/medqa_test.jsonl
python prepare_dataset.py --source pubmedqa --split test --limit 150 --out data/pubmedqa_test.jsonl
```
 
Open each output file and check the first 1-2 lines look sensible before
spending API budget on a full run.
 
## 2. The five modes
 
| Mode              | Legacy per-specialist tools | Conflict-directed retrieval | Devil's Advocate |
|-------------------|:---------------------------:|:----------------------------:|:-----------------:|
| `baseline`        | ✅ (unchanged original)      | ❌                            | ❌                 |
| `tools_off`       | ❌                            | ❌                            | ❌                 |
| `conflict_tools`  | ❌ (replaced)                 | ✅                            | ❌                 |
| `devils_advocate` | ✅ (unchanged original)      | ❌                            | ✅                 |
| `full`            | ❌ (replaced)                 | ✅                            | ✅                 |
 
`baseline` is your control group. `full` is your proposed system. The
other three isolate each contribution individually — useful for your
ablation table and for answering "was the improvement from A, B, or both?"
 
## 3. Sanity check first (do this before the full run)
 
```bash
python evaluate.py --dataset data/medqa_test.jsonl --mode baseline \
    --model gpt-4o-mini --output results/_sanity_check.jsonl --limit 5
```
 
Open `results/_sanity_check.jsonl` and confirm: final answers look
sensible, `rounds_used` is a small positive number, no `"error"` fields.
Fix anything broken here before scaling up — it's much cheaper to catch a
bug on 5 cases than on 150.
 
## 4. Run the full ablation
 
Repeat for each dataset and each mode (10 commands total per model):
 
```bash
python evaluate.py --dataset data/medqa_test.jsonl --mode baseline         --model gpt-4o-mini --output results/medqa_baseline_gpt4omini.jsonl
python evaluate.py --dataset data/medqa_test.jsonl --mode tools_off        --model gpt-4o-mini --output results/medqa_toolsoff_gpt4omini.jsonl
python evaluate.py --dataset data/medqa_test.jsonl --mode conflict_tools   --model gpt-4o-mini --output results/medqa_conflicttools_gpt4omini.jsonl
python evaluate.py --dataset data/medqa_test.jsonl --mode devils_advocate --model gpt-4o-mini --output results/medqa_devilsadvocate_gpt4omini.jsonl
python evaluate.py --dataset data/medqa_test.jsonl --mode full            --model gpt-4o-mini --output results/medqa_full_gpt4omini.jsonl
```
 
Then repeat the same 5 commands with `--dataset data/pubmedqa_test.jsonl`
(swap `medqa` for `pubmedqa` in the output filenames too).
 
## 5. Repeat on a second model (recommended: local Qwen2.5-7B via Ollama)
 
```bash
ollama pull qwen2.5:7b-instruct
ollama serve   # if not already running in the background
```
 
Then re-run the same 10 commands, swapping:
```
--model qwen2.5:7b-instruct --base_url http://localhost:11434/v1
```
and setting `MDT_API_KEY=ollama` (any non-empty placeholder string works
with Ollama's OpenAI-compatible endpoint).
 
## 6. Knowledge base — leave this off for the core ablation
 
By default `evaluate.py` runs with `use_kb=False` (CorrectKB/ChainKB
disabled). This is intentional: it isolates the effect of our two
contributions from the separate, confounding effect of experience
accumulation across cases. Do not pass `--use_kb` unless you're
deliberately running a follow-up experiment on that interaction — and if
you do, treat it as a separate experiment, not part of the core result.
 
## 7. After each batch finishes
 
Do not delete or overwrite `results/*.jsonl` files. These are your raw
evidence for the paper. Any analysis notebook should read from these files
only — never re-run an experiment "to fix a number."
 
## 8. What NOT to touch (agreed scope)
 
Do not modify `knowledge_base.py`'s `k=2` retrieval setting, do not
implement the paper's missing majority-vote/Reflector-for-deadlock logic,
do not change KB retrieval timing, and do not add few-shot exemplars to
the Primary Care Doctor prompt. These are known gaps between the paper and
the codebase, but they are not what our two contributions target — fixing
them would confound the comparison and is out of scope for this paper.
 
































