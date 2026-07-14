# MODIFIED for Devil's Advocate + Conflict-Directed Tool Retrieval.
# See EXPERIMENTS.md for what changed and why.
#
# Changes in this file:
#   - Two new sidebar checkboxes: "Enable Devil's Advocate" and
#     "Enable Conflict-Directed Tool Retrieval"
#   - MDTAgents() construction now passes those two flags through
#   - Initial state dict includes the new state fields workflow.py expects
#   - New UI block renders output from the new conflict_response_layer node

import streamlit as st
import base64
import json
from agents import MDTAgents
from workflow import create_workflow
from utils import load_config, save_config
from knowledge_base import kb_system

# Updated Page Config & Title
st.set_page_config(page_title="MDTeamGPT System", layout="wide", page_icon="🏥")

st.markdown("""
<style>
    .role-badge { background-color: #e8f4f8; padding: 4px 8px; border-radius: 4px; font-weight: bold; color: #0066cc; font-size: 0.9em;}
    .cot-box { border: 2px dashed #6f42c1; padding: 15px; border-radius: 10px; margin-top: 10px; background-color: #f3f0ff; }
    .retrieval-box { font-size: 0.85em; color: #555; border-left: 3px solid #6c757d; padding-left: 10px; margin-bottom: 5px; background: #fafafa; padding: 5px;}
    .tool-box { font-size: 0.85em; color: #2e7d32; border-left: 3px solid #2e7d32; padding-left: 10px; margin-bottom: 5px; background: #f1f8e9; padding: 5px;}
    .devil-box { font-size: 0.9em; color: #b71c1c; border-left: 3px solid #b71c1c; padding-left: 10px; margin-bottom: 5px; background: #fff3f3; padding: 8px; white-space: pre-wrap; }
    .context-label { font-weight: bold; color: #495057; font-size: 0.9em; }
    .saved-badge { color: green; font-weight: bold; }
    .not-saved-badge { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if "config" not in st.session_state:
    st.session_state.config = load_config()

# Sidebar
with st.sidebar:
    st.title("🏥 MDTeamGPT")
    st.caption("Multi-Agent Multidisciplinary Consultation System")

    with st.expander("⚙️ Connection Settings", expanded=False):
        with st.form("config_form"):
            api_key = st.text_input("API Key", value=st.session_state.config.get("api_key", ""), type="password")
            base_url = st.text_input("Base URL", value=st.session_state.config.get("base_url",
                                                                                   "https://dashscope.aliyuncs.com/compatible-mode/v1"))
            text_model = st.text_input("Text Model ID", value=st.session_state.config.get("text_model", "qwen-plus"))
            vl_model = st.text_input("Vision Model ID", value=st.session_state.config.get("vl_model", "qwen-vl-plus"))
            enable_tools = st.checkbox("Enable Internet/PubMed",
                                       value=st.session_state.config.get("enable_tools", True))
            if st.form_submit_button("Save Configuration"):
                new_conf = dict(st.session_state.config)
                new_conf.update({"api_key": api_key, "base_url": base_url, "text_model": text_model,
                                  "vl_model": vl_model, "enable_tools": enable_tools})
                save_config(new_conf)
                st.session_state.config = new_conf
                st.success("Configuration Saved!")

    # --- New: Experimental Features ---
    with st.expander("🔬 Experimental Features", expanded=True):
        enable_devils_advocate = st.checkbox(
            "Enable Devil's Advocate Agent",
            value=st.session_state.config.get("enable_devils_advocate", False),
            help="Adds a counter-argument agent that constructs the strongest "
                 "case for the minority position whenever the Lead Physician "
                 "detects a Conflict. The Safety Reviewer must weigh this "
                 "argument before declaring convergence."
        )
        enable_conflict_tools = st.checkbox(
            "Enable Conflict-Directed Tool Retrieval",
            value=st.session_state.config.get("enable_conflict_tools", False),
            help="Replaces the default per-specialist, every-round tool "
                 "calls with a single targeted search, fired only when a "
                 "Conflict is detected, built from the conflict text itself."
        )
        st.session_state.config["enable_devils_advocate"] = enable_devils_advocate
        st.session_state.config["enable_conflict_tools"] = enable_conflict_tools

    max_rounds = st.slider("Max Discussion Rounds", 3, 15, 6)

    st.divider()
    st.subheader("🧠 Context History")
    context_container = st.container()

# Main Interface
st.title("MDTeamGPT - Multi-Agent Multidisciplinary Consultation System")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Patient Data")
    case_input = st.text_area("Case Description", height=200, placeholder="Enter clinical details...")

    st.markdown("### 🎓 Training Mode")
    ground_truth = st.text_input("Ground Truth (Correct Answer)")

    img_file = st.file_uploader("Medical Image (Round 1 Only)", type=['jpg', 'png', 'jpeg'])
    img_base64 = None
    if img_file:
        st.image(img_file, caption="Uploaded Medical Scan", use_container_width=True)
        img_base64 = base64.b64encode(img_file.getvalue()).decode('utf-8')

    start_btn = st.button("🚀 Start Consultation", type="primary")


# UI Handler
class UIHandler:
    def __init__(self, container):
        self.root_container = container
        self.current_role = None
        self.role_expander = None
        self.text_placeholder = None
        self.full_text = ""

    def _ensure_expander(self, role):
        if role != self.current_role:
            self.current_role = role
            self.full_text = ""
            self.role_expander = self.root_container.expander(f"🗣️ {role} is speaking...", expanded=True)
            self.text_placeholder = self.role_expander.empty()

    def on_token(self, role, token):
        self._ensure_expander(role)
        self.full_text += token
        self.text_placeholder.markdown(self.full_text + "▌")

    def finish_turn(self):
        if self.text_placeholder:
            self.text_placeholder.markdown(self.full_text)

    def on_tool_output(self, role, query, result):
        self._ensure_expander(role)
        with self.role_expander:
            with st.expander(f"🛠️ Tool Usage: {query}", expanded=False):
                st.markdown(f"<div class='tool-box'>{result}</div>", unsafe_allow_html=True)


# Execution
if start_btn:
    cfg = st.session_state.config
    if not cfg.get("api_key"): st.stop()

    agents = MDTAgents(
        cfg["api_key"], cfg["base_url"], cfg["text_model"], cfg["vl_model"], cfg["enable_tools"],
        enable_conflict_tools=cfg.get("enable_conflict_tools", False),
        enable_devils_advocate=cfg.get("enable_devils_advocate", False),
    )
    app = create_workflow(agents)

    with col2:
        st.subheader("Consultation Process")
        status_log = st.status("Initializing Workflow...", expanded=True)
        chat_box = st.container()

        ui = UIHandler(chat_box)
        agents.set_stream_callback(ui.on_token)
        agents.set_tool_callback(ui.on_tool_output)

        state = {
            "case_info": case_input, "image_base64": img_base64, "ground_truth": ground_truth,
            "selected_roles": [], "triage_reason": "", "current_round": 1, "max_rounds": max_rounds,
            "context_bullets": [], "final_answer": "", "is_converged": False,
            "kb_context_text": "", "kb_context_docs": [],
            "use_knowledge_base": True,
            "pending_evidence": "", "current_devil_advocate_text": "",
            "round_conflict_flags": [], "devil_advocate_transcript": [], "tool_query_transcript": [],
        }

        try:
            for event in app.stream(state):

                if "triage" in event:
                    data = event["triage"]
                    status_log.write(f"✅ Triage Complete")

                    docs = data.get('kb_context_docs', [])
                    if docs:
                        with chat_box.expander(f"📚 Knowledge Retrieval ({len(docs)} Matches)", expanded=False):
                            for doc in docs:
                                source = doc.metadata.get("source_kb", "Unknown")
                                st.markdown(
                                    f"<div class='retrieval-box'><b>Source:</b> {source}<br>{doc.page_content}</div>",
                                    unsafe_allow_html=True)
                    else:
                        chat_box.caption("ℹ️ No relevant long-term experience found.")

                    chat_box.info(f"**📋 Triage Reasoning:** {data['triage_reason']}")
                    chat_box.success(f"**Selected Specialists:** {', '.join(data['selected_roles'])}")
                    chat_box.markdown("---")

                if "consultation_layer" in event:
                    ui.finish_turn()
                    data = event["consultation_layer"]
                    rnd = data["current_round"]
                    status_log.update(label=f"Round {rnd}: Consultation...", state="running")

                    # --- Update Sidebar with 6-Part Context ---
                    latest_bullet = data["context_bullets"][-1]
                    with context_container:
                        with st.expander(f"📝 Round {rnd} Context", expanded=False):
                            try:
                                ctx_data = json.loads(latest_bullet)
                                st.markdown(
                                    f"<span class='context-label'>Consistency:</span> {ctx_data.get('Consistency', '-')}",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    f"<span class='context-label'>Conflict:</span> {ctx_data.get('Conflict', '-')}",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    f"<span class='context-label'>Independence:</span> {ctx_data.get('Independence', '-')}",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    f"<span class='context-label'>Integration:</span> {ctx_data.get('Integration', '-')}",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    f"<span class='context-label'>Tools Usage:</span> {ctx_data.get('Tools_Usage', '-')}",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    f"<span class='context-label'>Long-Term Exp:</span> {ctx_data.get('Long_Term_Experience', '-')}",
                                    unsafe_allow_html=True)
                            except:
                                st.text(latest_bullet)

                # --- New: display output from the conflict_response_layer node ---
                if "conflict_response_layer" in event:
                    data = event["conflict_response_layer"]
                    tool_log = data.get("tool_query_transcript", [])
                    da_log = data.get("devil_advocate_transcript", [])

                    if tool_log:
                        with chat_box.expander("🎯 Conflict-Directed Retrieval Fired", expanded=False):
                            for entry in tool_log:
                                st.markdown(f"<div class='tool-box'>{entry}</div>", unsafe_allow_html=True)

                    if da_log:
                        with chat_box.expander("😈 Devil's Advocate Fired", expanded=True):
                            for entry in da_log:
                                st.markdown(f"<div class='devil-box'>{entry}</div>", unsafe_allow_html=True)

                if "safety_layer" in event:
                    data = event["safety_layer"]
                    if data["is_converged"]:
                        status_log.update(label="✅ Converged", state="complete", expanded=False)
                        st.balloons()
                        st.markdown("### 🏁 Final Medical Conclusion")
                        st.success(data["final_answer"])

                        # Training Logic
                        if ground_truth:
                            st.markdown("---")
                            st.markdown("### 🧪 Chain-of-Thought Review")
                            with st.spinner("Grading and saving experience..."):
                                result = agents.cot_reviewer(case_input, data["final_answer"], ground_truth)

                                st.markdown(f"<div class='cot-box'>", unsafe_allow_html=True)

                                if result.get("is_correct"):
                                    st.markdown("#### ✅ Answer is Correct")
                                    summary_s4 = result.get("summary_s4", "No summary provided.")
                                    st.write(f"**S4 Summary:** {summary_s4}")

                                    record = {
                                        "Question": case_input,
                                        "Answer": data["final_answer"],
                                        "Summary of S4_final": summary_s4
                                    }
                                    kb_system.save_correct_experience(record)

                                    st.markdown("---")
                                    st.markdown(f"<span class='saved-badge'>✅ Saved to: CorrectKB</span>",
                                                unsafe_allow_html=True)
                                    st.markdown(
                                        f"<span class='not-saved-badge'>❌ NOT Saved to: ChainKB (Reason: Answer was correct)</span>",
                                        unsafe_allow_html=True)

                                else:
                                    st.markdown("#### ❌ Answer is Incorrect")
                                    st.write(f"**Error Reflection:** {result.get('error_reflection', '-')}")

                                    record = {
                                        "Question": case_input,
                                        "Correct Answer": ground_truth,
                                        "Initial Hypothesis": result.get("initial_hypothesis", "-"),
                                        "Analysis Process": result.get("analysis_process", "-"),
                                        "Final Conclusion": result.get("final_conclusion", "-"),
                                        "Error Reflection": result.get("error_reflection", "-")
                                    }
                                    kb_system.save_reflection_experience(record)

                                    st.markdown("---")
                                    st.markdown(f"<span class='saved-badge'>✅ Saved to: ChainKB</span>",
                                                unsafe_allow_html=True)
                                    st.markdown(
                                        f"<span class='not-saved-badge'>❌ NOT Saved to: CorrectKB (Reason: Answer was incorrect)</span>",
                                        unsafe_allow_html=True)

                                st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        chat_box.warning("⚠️ Divergence detected. Continuing...")

        except Exception as e:
            st.error(f"Error: {e}")