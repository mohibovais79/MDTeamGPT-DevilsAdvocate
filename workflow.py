# triage -> consultation_layer -> conflict_response_layer -> safety_layer -> (loop or end)

# `conflict_response_layer` is the new node. Every round, right after the
# Lead Physician's synthesis is produced, it:
#   1. Parses the "Conflict" field out of that synthesis.
#   2. If (and only if) a genuine conflict is detected:
#        a. (if enable_conflict_tools) fires ONE targeted tool query built
#           from the conflict text, storing the result as `pending_evidence`
#           for the NEXT round's specialist discussion.
#        b. (if enable_devils_advocate) runs the Devil's Advocate agent,
#           whose counter-argument is passed to the Safety Reviewer THIS
#           round before it decides CONVERGED / DIVERGED.
#
# Both behaviors are OFF by defaul.



from typing import TypedDict, List, Annotated, Any
import json
import operator
from langgraph.graph import StateGraph, END
from knowledge_base import kb_system


class MDTState(TypedDict):
    case_info: str
    image_base64: str
    ground_truth: str

    selected_roles: List[str]
    triage_reason: str

    current_round: int
    max_rounds: int

    context_bullets: Annotated[List[str], operator.add]
    final_answer: str
    is_converged: bool

    kb_context_text: str
    kb_context_docs: Any

    use_knowledge_base: bool

# `pending_evidence` is produced by conflict_response_layer in round N
    # (based on round N's detected conflict) and consumed by
    # consultation_layer at the START of round N+1.
    pending_evidence: str
    # `current_devil_advocate_text` is produced and consumed within the SAME
    # round (conflict_response_layer -> safety_layer), then reset to "".
    current_devil_advocate_text: str
 
  
    round_conflict_flags: Annotated[List[bool], operator.add]
    devil_advocate_transcript: Annotated[List[str], operator.add]
    tool_query_transcript: Annotated[List[str], operator.add]
 



#  helpers for parsing the Lead Physician's JSON synthesis and
# deciding whether it represents a genuine, actionable conflict. ---
 
def _safe_parse_lead_json(raw: str) -> dict:
    """Parses the Lead Physician's JSON synthesis defensively. Returns {}
    on any parse failure rather than raising, so a single malformed LLM
    output cannot crash an entire evaluation run."""
    try:
        return json.loads(raw)
    except Exception:
        return {}
 
 
_NO_CONFLICT_PHRASES = {
    "", "none", "n/a", "na", "no conflict", "no conflicts",
    "no conflicts identified", "no conflicts identified.",
    "no significant conflict", "no major conflicts", "none.",
    "no conflict identified", "no conflict identified.",
}
 
 
def _has_real_conflict(conflict_text: str) -> bool:
    """The Lead Physician's prompt asks for an empty Conflict field when
    there is none, but LLMs frequently emit phrases like "None" or "No
    conflicts identified" instead of a literal empty string. This
    normalizes those cases so they are correctly treated as "no conflict"
    rather than accidentally triggering our new mechanisms on every case."""
    if not conflict_text:
        return False
    normalized = conflict_text.strip().lower().rstrip(".")
    return normalized not in _NO_CONFLICT_PHRASES


def create_workflow(agents_instance):
    def node_triage(state: MDTState):
        kb_system.init_embeddings(
            api_key=agents_instance.llm.openai_api_key,
            base_url=agents_instance.llm.openai_api_base
        )

        retrieval_result = kb_system.retrieve_context_details(state["case_info"])
        triage_result = agents_instance.primary_care_doctor(state["case_info"])

        return {
            "selected_roles": triage_result["selected_roles"],
            "triage_reason": triage_result["reasoning"],
            "current_round": 1,
            "kb_context_text": retrieval_result["text"],
            "kb_context_docs": retrieval_result["docs"],
            "context_bullets": [],
            "pending_evidence": "",
            "current_devil_advocate_text": "",
            "round_conflict_flags": [],
            "devil_advocate_transcript": [],
            "tool_query_transcript": [],
        }

    def node_consultation_and_synthesis(state: MDTState):
        roles = state["selected_roles"]
        rnd = state["current_round"]
        bullets = state["context_bullets"]

        #  Logic Check: Residual Context
        # 1. This is calculated BEFORE the agent loop.
        # 2. It only contains info from PREVIOUS rounds (bullets).
        # 3. Therefore, agents in this round CANNOT see each other's current output.
        residual_context = ""
        if rnd == 1:
            residual_context = f"PRIOR KNOWLEDGE FROM DB:\n{state['kb_context_text']}"
        else:
            recent_bullets = bullets[-2:]
            for i, b in enumerate(recent_bullets):
                bullet_rnd = rnd - len(recent_bullets) + i
                residual_context += f"--- Round {bullet_rnd} Summary ---\n{b}\n"

        injected_evidence = state.get("pending_evidence", "")
        dialogues = []
        for role in roles:
            img = state["image_base64"] if rnd == 1 else None

            # Logic Check: Independence & Blindness
            # 1. 'residual_context' is static for all agents in this loop.
            # 2. 'ground_truth' is NOT passed to the agent.
            res = agents_instance.specialist_consult(
                role, state["case_info"], residual_context, img, rnd, injected_evidence=injected_evidence
            )
            dialogues.append(f"**{role}**: {res}")

        # Lead Physician synthesizes the accumulated dialogues
        summary_json = agents_instance.lead_physician_synthesis(dialogues, rnd)

        return {
            "context_bullets": [summary_json],
            "current_round": rnd
        }

    def node_conflict_response(state: MDTState):
        """
        Runs once per round, right after the Lead Physician's
        synthesis, and before the Safety Reviewer's convergence check.
 
        Detects whether this round's synthesis contains a genuine conflict,
        and if so, activates whichever of our two contributions are enabled:
        conflict-directed tool retrieval, and/or the Devil's Advocate agent.
        """
        rnd = state["current_round"]
        last_bullet = state["context_bullets"][-1]
 
        parsed = _safe_parse_lead_json(last_bullet)
        conflict_text = parsed.get("Conflict", "")
        conflict_detected = _has_real_conflict(conflict_text)
 
        new_evidence = ""
        tool_log_entry = ""
        if conflict_detected and agents_instance.enable_conflict_tools:
            query = agents_instance.generate_conflict_query(conflict_text)
            if query and "no query" not in query.lower():
                new_evidence = agents_instance.tools.run_tools(query)
                tool_log_entry = f"Round {rnd} | Query: {query}\nResult: {new_evidence}"
 
        da_text = ""
        if conflict_detected and agents_instance.enable_devils_advocate:
            da_text = agents_instance.devil_advocate_argument(
                last_bullet, new_evidence, rnd
            )
 
        return {
            "pending_evidence": new_evidence,
            "current_devil_advocate_text": da_text,
            "round_conflict_flags": [conflict_detected],
            "devil_advocate_transcript": [da_text] if da_text else [],
            "tool_query_transcript": [tool_log_entry] if tool_log_entry else [],
        }

    def node_safety_check(state: MDTState):
        last_bullet = state["context_bullets"][-1]
        rnd = state["current_round"]
        devil_advocate_text = state["current_devil_advocate_text"]

        # Safety Reviewer checks convergence based on the summary
        review = agents_instance.safety_reviewer(last_bullet, rnd, devil_advocate_text=devil_advocate_text)

        is_converged = "STATUS: CONVERGED" in review
        final_ans = ""

        if "FINAL_ANSWER:" in review:
            parts = review.split("FINAL_ANSWER:")
            final_ans = parts[1].strip() if len(parts) > 1 else review

        if rnd >= state["max_rounds"]:
            is_converged = True
            if not final_ans:
                final_ans = "Max rounds reached. Proceeding with latest hypothesis."

        return {
            "is_converged": is_converged,
            "final_answer": final_ans,
            "current_round": rnd + 1,
            "current_devil_advocate_text": ""
        }

    def router(state: MDTState):
        if state["is_converged"]:
            return "end"
        return "continue"

    workflow = StateGraph(MDTState)

    workflow.add_node("triage", node_triage)
    workflow.add_node("consultation_layer", node_consultation_and_synthesis)
    workflow.add_node("conflict_response_layer", node_conflict_response)
    workflow.add_node("safety_layer", node_safety_check)
 
    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "consultation_layer")
    workflow.add_edge("consultation_layer", "conflict_response_layer")
    workflow.add_edge("conflict_response_layer", "safety_layer")
 
    workflow.add_conditional_edges("safety_layer", router, {"continue": "consultation_layer", "end": END})

    return workflow.compile()