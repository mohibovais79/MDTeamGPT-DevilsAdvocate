# MDTeamGPT Extension — Research Context & Sync Notes

## 1. Base Paper

**Title:** MDTeamGPT: Mitigating Context Collapse and Enabling Self-Evolution in Medical Multi-Agent Reasoning  
**Venue:** ACL 2026 Findings (pages 28578–28606)  
**Authors:** Kai Chen et al., Nanjing University  
**Code:** This repository (`d:\MDTeamGPT`)

### Paper's Two Core Contributions
1. **Residual Discussion Structure** — Lead Physician + sliding window of 2 previous round summaries to mitigate context collapse
2. **Dual Knowledge Base** — CorrectKB (successful reasoning) + ChainKB (error reflections) for self-evolution via FAISS

---

## 2. Target Extension Paper

**Target Venue:** FIT 2026 (23rd International Conference on Frontiers of Information Technology), COMSATS University Islamabad, December 14–15 2026. IEEE Xplore indexed.  
**Target Tracks:** "Human-Centered Agentic AI" or "Natural Language Processing & Generative AI"  
**Submission:** EasyChair at https://easychair.org/conferences/?conf=fit26  
**Format:** 6-page IEEE format

### Proposed Title
"Mitigating Anchoring Bias in Multi-Agent Medical Consultation via Devil's Advocate Reasoning and Conflict-Directed Evidence Retrieval"

---

## 3. Our Two Contributions

### Contribution 1: Anchoring Bias Detection + Devil's Advocate

**Problem:** The paper documents a failure in Appendix B.2 (Takayasu arteritis case) where 3/5 agents anchored on neurological symptoms across 15 rounds and never explored cardiovascular alternatives. The paper's own error analysis says: *"The case design cleverly exploited the anchoring effect in clinical thinking."* The framework has NO mechanism to distinguish genuine consensus from anchored collective convergence.

**Mechanism:** A new `anchoring_detector` agent that runs after round 2. It inspects the `Consistency` field of the last two residual context summaries. If agents cite the **same symptom cluster** as primary evidence across both rounds without exploring alternatives, it flags anchoring. When flagged, one specialist is re-prompted as **devil's advocate** — forced to argue the strongest alternative hypothesis.

**Where in code:**
- New method in `agents.py`: `devil_advocate_argument()`
- New condition in `workflow.py` inside `node_consultation_and_synthesis`
- Controlled by flag: `self.enable_devils_advocate` (added to `MDTAgents.__init__`)

**Motivation source:** Author-documented failure (Appendix B.2), NOT inferred

### Contribution 2: Conflict-Triggered Tool Invocation

**Problem:** The paper's Section 6 (Limitations) explicitly states future work should focus on *"deeply integrating specialized medical tools for collaborative reasoning."* Currently tools fire blindly every round for every specialist (`agents.py:104-119`) based on raw case text — they never respond to detected conflict. The `Conflict` field in Lead Physician JSON is computed but never acted on programmatically.

**Mechanism:** After Lead Physician synthesis, parse the `Conflict` field. If non-empty (genuine disagreement), run `MedicalTools.run_tools()` with a targeted query built from the conflicting hypotheses. Inject results into the **next round's residual context** as `Forced_Evidence`. Suppress the old blind per-specialist tool firing when this mode is active.

**Where in code:**
- New method in `agents.py`: `generate_conflict_query()`
- New node/logic in `workflow.py`: `conflict_response_layer` between Lead Physician output and next round
- `specialist_consult()` tool block gated behind `self.enable_conflict_tools`
- Controlled by flag: `self.enable_conflict_tools` (added to `MDTAgents.__init__`)

**Motivation source:** Author-stated limitation (Section 6) + author-documented tool absence in failure case (Appendix B.2: *"Tools Usage: None. The absence of external tool verification... contributed to the failure"*)

---

## 4. Paper vs Code Gaps (Verified by Full Audit)

### Missing from code (paper describes but code doesn't implement):
- **Majority vote** for closed-ended MCQ (Algorithm 1 line 31) — Safety Reviewer LLM decides instead
- **Reflector agent** for open-ended deadlock — max rounds just outputs generic message
- **KB retrieval deferred to Round 2** (paper Section 3.3) — code injects KB at Round 1
- **Few-shot exemplars** for Primary Care Doctor (paper Section 3.1) — not in prompt
- **K=5 retrieval** (paper Appendix C.9) — code uses K=2 in `knowledge_base.py`

### Fix before experiments:
- Change K=2 → K=5 in `knowledge_base.py` to match paper's validated parameter

---

## 5. Tool Call Behavior (Verified)

Current tool triggering in `agents.py:104-119`:
- Fires in **ALL rounds** (not Round 1 only)
- Fires for **every specialist** unconditionally
- Uses `critic_llm` (temperature=0) to generate keyword from first 300 chars of case
- Only skipped if `enable_tools=False` in config or LLM returns "no query"
- Runs **both** DuckDuckGo AND PubMed with same query
- Results truncated to 600 chars each
- Happens **before** Lead Physician synthesis → Conflict field cannot influence tool calls

**For ablation:** Need toggle to switch between blind per-specialist tool firing (baseline) vs conflict-triggered framework firing (our contribution).

---

## 6. Experimental Plan

### Models
| Model | Provider | Cost | Comparable to Paper? |
|---|---|---|---|
| DeepSeek V3 | api.deepseek.com ($0.01/$0.03 per 1M) | ~$3 for full run | ✅ Yes (Figure 4) |
| Qwen2.5-14B | Local ollama on Mac Mini M4 16GB | $0 | ✅ Qwen lineage |
| llama3-8B | Local ollama on RTX 5060 8GB | $0 | ✅ Yes (Figure 4) |

### Datasets
- **MedQA** (USMLE): 1,273 test cases — primary benchmark
- **PubMedQA**: 500 test cases — secondary benchmark
- **MedQA-Conflict subset**: filtered from MedQA full run (cases with non-empty Conflict field)
- Source: `bigbio/med_qa` on HuggingFace or https://github.com/jind11/MedQA

### KB Training
- Build from ~300 MedQA training cases (paper used 900, self-evolution curve plateaus at ~300)
- Acknowledge difference in paper limitations

### Experiment Configurations (4 configs × 2 datasets × 3 models)
1. Base MDTeamGPT (no changes)
2. + Contribution 1 only (anchoring detector + devil's advocate)
3. + Contribution 2 only (conflict-triggered tools)
4. Both contributions (full extended system)

### Estimated API cost: ~$38 total (well within $100-200 budget)

---

## 7. Team Division (2 people)

### Person A — Systems & Experiments
- Environment setup, code implementation, running experiments, logging results
- Maintains `results_log.md` in repo

### Person B — Literature & Paper Structure
- Related work (anchoring bias + tool-augmented agents), paper skeleton, tables, figures, writing

### Timeline
- **Week 1:** Joint setup + first demo run
- **Week 2:** A: baseline reproduction | B: related work + paper skeleton
- **Week 3:** A: KB training + implement Contribution 2 | B: design Contribution 1 prompt + write Method section
- **Week 4:** A: full 4-config evaluation | B: build tables + analyze results
- **Week 5:** A: technical writing | B: intro + conclusion
- **Week 6:** Joint polish + submission

---

## 8. Paper Structure (6 pages IEEE)

```
1. Introduction          (~1 page)   — anchoring bias problem + conflict-triggered gap
2. Related Work          (~0.5 page) — cite MDTeamGPT + anchoring bias literature
3. Method                (~1.5 pages)— two mechanisms + system diagram
4. Experiments           (~2 pages)  — 3 experiment types + ablation table
5. Conclusion            (~0.25 page)
References               (~0.75 page)
```

### Three Experiment Types
1. **Motivating case (qualitative):** Takayasu arteritis from Appendix B.2 — show mechanisms activate
2. **Conflict-heavy subset (quantitative):** MedQA-Conflict subset comparison across 4 configs
3. **Ablation:** Table mirroring original paper's Table 2 format

---

## 9. Current Implementation Progress

### Done:
- Docstring added to `agents.py` describing new architecture
- `enable_conflict_tools` and `enable_devils_advocate` flags added to `MDTAgents.__init__`
- KB `save_correct_experience` and `save_reflection_experience` guarded against uninitialized state

### TODO:
- [ ] Add `enable_conflict_tools` and `enable_devils_advocate` params to `__init__` signature
- [ ] Implement `generate_conflict_query()` method in `agents.py`
- [ ] Implement `devil_advocate_argument()` method in `agents.py`
- [ ] Gate existing tool block in `specialist_consult()` behind `enable_conflict_tools`
- [ ] Add `conflict_response_layer` logic in `workflow.py`
- [ ] Add anchoring detection logic in `workflow.py`
- [ ] Update `app.py` UI with toggles for new flags
- [ ] Fix K=2 → K=5 in `knowledge_base.py`
- [ ] Build dataset loading/evaluation scripts
- [ ] Create `results_log.md` template

---

## 10. Key Decisions Made

1. **No governance layer** — deemed redundant with existing system, not a research contribution
2. **No image/VLM work** — focusing on text-only experiments
3. **DeepSeek V3 as primary API model** — cheapest strong model, already in paper's evaluation
4. **Open-source local models supplement** — llama3-8B + qwen2.5-14B for reproducibility angle
5. **KB training with 300 cases** — not full 900, justified by plateau in Figure 4
6. **Two contributions target different failure modes** — false consensus (C1) vs unresolved conflict (C2)
