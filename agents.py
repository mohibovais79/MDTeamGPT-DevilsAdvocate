# MODIFIED for Devil's Advocate + Conflict-Directed Tool Retrieval, PLUS a
# provider switch to properly support local Ollama reasoning models.
# See EXPERIMENTS.md for the ablation-related changes.
#
# --- Provider switch (new) ---
# ChatOpenAI (used for any real OpenAI-compatible API: OpenAI, Groq,
# DashScope) has its OWN "reasoning" field meant for OpenAI's o1/o3 models,
# expecting a dict like {"effort": "low"} -- passing a boolean through
# model_kwargs collides with that field and raises a pydantic ValidationError.
#
# ChatOllama (from the separate `langchain_ollama` package) talks to
# Ollama's NATIVE API directly and has proper first-class support for
# `reasoning=True/False`, which fully disables "thinking" mode on reasoning
# models (e.g. Qwen3.5, DeepSeek-R1) -- this is what actually fixes the
# 2:28-minute-per-call slowdown caused by thinking traces.
#
# Set provider="ollama" when running against a local Ollama server.
# Requires: uv add langchain-ollama   (or: pip install langchain-ollama)
#
# Two new agent methods added:
#   - generate_conflict_query()   -> builds ONE targeted search query from a
#                                    detected Lead-Physician "Conflict" field
#   - devil_advocate_argument()   -> constructs the strongest case for the
#                                    minority/alternative position
#
# One existing method modified:
#   - specialist_consult()  -> the old blind, per-specialist, every-round
#                              tool call is now gated behind
#                              `enable_conflict_tools`. When conflict-directed
#                              retrieval is active, specialists no longer
#                              fire their own independent searches; instead
#                              they receive `injected_evidence` gathered by
#                              the new conflict_response_layer (see
#                              workflow.py) from the PREVIOUS round.


from typing import List, Dict, Any, Callable
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tools import MedicalTools

SPECIALIST_POOL = [
    "General Internal Medicine Doctor",
    "General Surgeon",
    "Pediatrician",
    "Obstetrician and Gynecologist",
    "Radiologist",
    "Neurologist",
    "Pathologist",
    "Pharmacist"
]


class MDTAgents:
    def __init__(self, api_key, base_url, text_model, vl_model, enable_tools=True,
                 enable_conflict_tools=False, enable_devils_advocate=False,
                 provider="openai"):
        """
        provider:
            "openai" (default) -- any OpenAI-compatible API (OpenAI, Groq,
                DashScope, etc). Uses langchain_openai.ChatOpenAI.
            "ollama" -- a locally-served Ollama model. Uses
                langchain_ollama.ChatOllama with reasoning=False, which is
                the ONLY reliable way to disable "thinking" mode on
                reasoning models served through Ollama. Requires
                `uv add langchain-ollama` (or pip install langchain-ollama).

                base_url for this mode should be the Ollama server root
                (e.g. "http://localhost:11434"), NOT the OpenAI-compat
                "/v1" path -- but if you pass the "/v1" form out of habit,
                it is stripped automatically below.
        """
        self.provider = provider

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            ollama_base_url = base_url[:-3] if base_url.endswith("/v1") else base_url

            self.llm = ChatOllama(
                model=text_model,
                base_url=ollama_base_url,
                temperature=0.7,
                reasoning=False,
            )
            self.critic_llm = ChatOllama(
                model=text_model,
                base_url=ollama_base_url,
                temperature=0.0,
                reasoning=False,
            )
            self.vl_llm = ChatOllama(
                model=vl_model,
                base_url=ollama_base_url,
                temperature=0.1,
                num_predict=2048,
                reasoning=False,
            )
        else:
            self.llm = ChatOpenAI(
                model=text_model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.7,
                streaming=True
            )
            self.critic_llm = ChatOpenAI(
                model=text_model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.0,
                streaming=False
            )
            self.vl_llm = ChatOpenAI(
                model=vl_model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.1,
                max_tokens=2048,
                streaming=True
            )

        self.tools = MedicalTools(enable=enable_tools)

        # --- Experimental condition toggles (Devil's Advocate / Conflict
        # tools). Both default to False so constructing MDTAgents with no
        # extra arguments reproduces the ORIGINAL, unmodified behavior
        # exactly -- this is your "baseline" ablation condition.
        self.enable_conflict_tools = enable_conflict_tools
        self.enable_devils_advocate = enable_devils_advocate

        # Callbacks
        self.stream_callback = None
        self.tool_callback = None

    def set_stream_callback(self, callback: Callable[[str, str], None]):
        self.stream_callback = callback

    def set_tool_callback(self, callback: Callable[[str, str, str], None]):
        self.tool_callback = callback

    # 1. Primary Care (Triage)
    def primary_care_doctor(self, case_info: str) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_template(
            """You are a Primary Care Doctor at the Triage Desk.
            Analyze the patient case and select the most appropriate specialists.

            Available Specialists:
            {pool}

            Patient Case: {case}

            TASK:
            1. Explain your reasoning.
            2. Select AT LEAST 3 specialists.

            OUTPUT JSON FORMAT:
            {{
                "reasoning": "...",
                "selected_roles": ["Role A", "Role B", "Role C"]
            }}
            """
        )
        chain = prompt | self.llm
        result = chain.invoke({"pool": ", ".join(SPECIALIST_POOL), "case": case_info})

        content = result.content.strip()
        if content.startswith("```json"): content = content[7:]
        if content.endswith("```"): content = content[:-3]

        try:
            data = json.loads(content)
            selected = [s for s in data.get("selected_roles", []) if s in SPECIALIST_POOL]
            remaining = [s for s in SPECIALIST_POOL if s not in selected]
            while len(selected) < 3 and remaining:
                selected.append(remaining.pop(0))
            data["selected_roles"] = selected
            return data
        except:
            return {
                "reasoning": "Fallback selection.",
                "selected_roles": ["General Internal Medicine Doctor", "General Surgeon", "Radiologist"]
            }

    #2. Specialists (Consultation)
    def specialist_consult(self, role: str, case_info: str, residual_context: str,
                           image_data=None, round_num=1, injected_evidence: str = ""):

        #Tool Usage Logic
        tool_context = ""
        if self.tools.enable and not self.enable_conflict_tools:
            try:
                kw_prompt = ChatPromptTemplate.from_template(
                    "Extract 1 specific medical query string for {role} to research regarding: {case}. Return ONLY the query.")
                kw_chain = kw_prompt | self.critic_llm
                kw = kw_chain.invoke({"case": case_info[:300], "role": role}).content

                if kw and "no query" not in kw.lower():
                    tool_res = self.tools.run_tools(kw)
                    if tool_res:
                        if self.tool_callback:
                            self.tool_callback(role, kw, tool_res)
                        tool_context = f"\n[External Tools Data]:\n{tool_res}\n"
            except Exception as e:
                print(f"Tool error: {e}")

        if injected_evidence:
            tool_context += (
                f"\n[Targeted Evidence — retrieved after the previous round's "
                f"detected disagreement]:\n{injected_evidence}\n"
            )

        

        # Strict Reasoning Structure
        structure_instruction = """
        IMPORTANT INSTRUCTIONS:
        1. **Independence**: You are providing your opinion INDEPENDENTLY. You cannot see the opinions of other specialists in this current round. You can only see the summary of previous rounds (if any).
        2. **Blindness**: You do NOT have access to the ground truth or final correct diagnosis. Rely only on the case description and your knowledge.
        3. **Structure**: You must structure your response in exactly three sections:

           - **1. Context Summary**: 
             (If Round 1: Summarize "Prior Knowledge". If Round > 1: Summarize "Residual Context" from previous rounds.)

           - **2. Clinical Reasoning**: 
             (Analyze the case. If tool data exists, use it. If image exists, describe findings. Explain step-by-step.)

           - **3. Conclusion**: 
             (State your clear medical opinion or diagnosis.)

           - **4. Choice**: 
             (If the case provides multiple-choice options A-E, you MUST end with a line in this exact format:)
             Choice: {Option ID}: {Option Content}
             (Example: Choice: D: Blockade of presynaptic acetylcholine release at the neuromuscular junction)
             (If no multiple-choice options are provided, omit this section.)
        """

        system_prompt = f"You are a {role}. Provide expert medical opinion.\n{structure_instruction}"

        user_text = f"Patient Case: {case_info}\n{tool_context}\n"

        if round_num == 1:
            user_text += "\n[Status]: Round 1. Analyze independently."
            user_text += f"\n*** PRIOR KNOWLEDGE / CONTEXT ***\n{residual_context}\n"
            if image_data:
                user_text += " [Image Provided]. Describe findings and integrate with diagnosis."
            else:
                user_text += " No image provided."
        else:
            user_text += f"\n[Status]: Round {round_num}.\n"
            user_text += f"*** RESIDUAL CONTEXT (Previous Rounds) ***\n{residual_context}\n"
            user_text += "Review the summaries of previous rounds. Support, refute, or synthesize based on that history."

        messages = [SystemMessage(content=system_prompt)]

        target_llm = self.llm
        if round_num == 1 and image_data:
            target_llm = self.vl_llm
            img_url = f"data:image/jpeg;base64,{image_data}"
            content_payload = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]
            messages.append(HumanMessage(content=content_payload))
        else:
            messages.append(HumanMessage(content=user_text))

        try:
            full_res = ""
            for chunk in target_llm.stream(messages):
                token = chunk.content
                full_res += token
                if self.stream_callback: self.stream_callback(role, token)
            return full_res
        except Exception as e:
            return f"Error: {e}"

    #3. Lead Physician
    def lead_physician_synthesis(self, round_dialogues: List[str], round_num: int):
        # Lead Physician DOES see all dialogues from the current round (to synthesize them),
        # but DOES NOT see Ground Truth.
        prompt = ChatPromptTemplate.from_template(
            """You are the Lead Physician.
            Synthesize the specialists' discussions from Round {rnd} into a concise structured summary.

            Specialists' Output (Current Round):
            {dialogues}

            TASK:
            Create a JSON object containing EXACTLY these 7 fields:

            1. "Consistency": (Aggregates the parts of individual statements that are consistent across multiple agent statements).
            2. "Conflict": (Identifies conflicting points between statements; empty if none).
            3. "Independence": (Extracts unique viewpoints of each agent not mentioned by others).
            4. "Integration": (Synthesizes all statements into a cohesive summary).
            5. "Tools_Usage": (Summarize specific tools/searches used in this round).
            6. "Long_Term_Experience": (Extract and summarize any prior experience/knowledge referenced from the database).
            7. "has_conflict": (boolean: true if the specialists disagree on any diagnostic or clinical point, false if they are in agreement).

            Return ONLY valid JSON.
            """
        )
        chain = prompt | self.llm
        res = chain.invoke({
            "rnd": round_num,
            "dialogues": "\n\n".join(round_dialogues)
        })

        content = res.content.strip()
        if content.startswith("```json"): content = content[7:]
        if content.endswith("```"): content = content[:-3]
        return content.strip()
    

    # Conflict-Directed Query Generation (Contribution 2)
    def generate_conflict_query(self, conflict_text: str) -> str:
        """
        Given the Lead Physician's "Conflict" field for this round, generate
        ONE specific, targeted medical literature search query aimed at
        resolving that specific disagreement. This replaces the old
        per-specialist, per-round, generic-case-text query with a single
        query built directly from what the team is actually disputing.
        """
        prompt = ChatPromptTemplate.from_template(
            """You are assisting a medical team that has an unresolved disagreement.
 
            CONFLICT DESCRIPTION:
            {conflict}
 
            TASK:
            Generate ONE specific, targeted medical literature search query
            that would help resolve this specific disagreement (e.g. comparing
            the competing diagnoses directly, checking complication rates, or
            verifying a specific clinical fact in dispute). Return ONLY the
            query text, nothing else. If the conflict text is not resolvable
            by literature search, return "no query".
            """
        )
        chain = prompt | self.critic_llm
        res = chain.invoke({"conflict": conflict_text})
        return res.content.strip()
 
    #  Devil's Advocate Agent (Contribution 1)
    def devil_advocate_argument(self, lead_summary_json: str, tool_evidence: str,
                                 round_num: int) -> str:
        """
        Constructs the strongest possible case for the minority/alternative
        position identified in this round's Lead Physician synthesis. This
        is only invoked when the Lead Physician's "Conflict" field is
        non-empty (see workflow.py: node_conflict_response), so it never
        fires on easy, unanimous cases.
 
        """
        evidence_section = ""
        if tool_evidence:
            evidence_section = (
                f"\nEXTERNAL EVIDENCE RETRIEVED (based on the detected "
                f"disagreement):\n{tool_evidence}\n"
            )
 
        prompt = ChatPromptTemplate.from_template(
            """You are the Devil's Advocate on this Multi-Disciplinary Team.
 
            Your role is NOT to give your own diagnosis. Your role is to
            identify the position that is currently the MINORITY or
            LESS-SUPPORTED view in the Lead Physician's synthesis below, and
            construct the strongest possible clinical argument FOR that
            position — even if you personally suspect the majority is
            correct.
 
            This is a structural safeguard against groupthink and anchoring
            bias: teams sometimes converge quickly on a plausible-sounding
            diagnosis while overlooking a less obvious but more dangerous
            alternative.
 
            ROUND {rnd} — LEAD PHYSICIAN'S SYNTHESIS
            (Consistency / Conflict / Independence / Integration / Tools / Memory):
            {lead_summary}
            {evidence_section}
 
            TASK:
            1. Identify the CURRENT MAJORITY / LEADING position from the synthesis.
            2. Identify the MINORITY / ALTERNATIVE position(s) mentioned in
               the "Conflict" or "Independence" fields.
            3. Construct the strongest possible clinical argument FOR the
               minority position, citing specific clinical reasoning and any
               evidence provided above.
            4. State what would need to be true, or what evidence would need
               to exist, for the minority position to be correct instead of
               the majority.
 
            OUTPUT FORMAT (strict):
            MAJORITY_POSITION: [...]
            MINORITY_POSITION: [...]
            COUNTER_ARGUMENT: [...]
            WHAT_WOULD_CONFIRM_MINORITY: [...]
            """
        )
        chain = prompt | self.llm
        res = chain.invoke({
            "rnd": round_num,
            "lead_summary": lead_summary_json,
            "evidence_section": evidence_section,
        })
        return res.content

    #4. Safety Reviewer
    def safety_reviewer(self, current_bullet: str, round_num: int,
                         devil_advocate_text: str = "", case_info: str = ""):

        da_section=""

        if devil_advocate_text:
            da_section = (
                f"\n\nDEVIL'S ADVOCATE COUNTER-ARGUMENT (you must weigh this "
                f"before declaring convergence):\n{devil_advocate_text}\n"
            )


        prompt = ChatPromptTemplate.from_template(
            """You are the Safety and Ethics Reviewer.
            Review the current round's synthesis.

            Patient Case (includes multiple-choice options if applicable):
            {case}

            Current Round Synthesis:
            {bullet}

            {da_section}

            TASK:
            Determine if the medical diagnosis has converged to a solid, safe conclusion without major conflicts.

            If a Devil's Advocate counter-argument is present above, you must
            explicitly weigh it in your REASON — do not declare CONVERGED if
            the counter-argument raises a substantive, unaddressed clinical
            concern that the team has not actually resolved.

            If the case has answer options listed above, you MUST identify
            which option KEY the team's consensus points to and put that key
            in the "choice" field. Use the EXACT key as shown in the options:
            - If options are labeled A, B, C, D, E -> put the letter (e.g. "D")
            - If options are yes/no/maybe -> put "yes", "no", or "maybe"
            Map the diagnosis or conclusion text from the synthesis to the
            correct option key.

            OUTPUT FORMAT (JSON only, no markdown fences):
            {{
              "status": "CONVERGED" or "DIVERGED",
              "reason": "Short explanation",
              "choice": "The option key that matches the team's consensus (e.g. A, B, C, D, E, yes, no, or maybe). Empty string only if not converged or no options exist."
            }}
            """
        )
        chain = prompt | self.critic_llm
        res = chain.invoke({"bullet": current_bullet, "da_section": da_section, "case": case_info})
        content = res.content.strip()
        if content.startswith("```json"): content = content[7:]
        if content.startswith("```"): content = content[3:]
        if content.endswith("```"): content = content[:-3]
        try:
            return json.loads(content)
        except Exception:
            return {"status": "DIVERGED", "reason": "Parse error", "choice": ""}

    # 5. CoT Reviewer
    def cot_reviewer(self, case_info, final_answer, ground_truth):
        # Only this agent sees the Ground Truth
        prompt = ChatPromptTemplate.from_template(
            """You are the 'Chain-of-Thought Reviewer'.

            CASE: {case}
            MODEL ANSWER: {answer}
            GROUND TRUTH: {truth}

            TASK:
            Step 1: Determine correctness (letters match for Choice, semantic match for Open).

            Step 2: Generate specific fields based on correctness.

            IF CORRECT:
               - "is_correct": true
               - "summary_s4": A concise summary of the final reasoning (S4_final).

            IF INCORRECT:
               - "is_correct": false
               - "initial_hypothesis": What was the likely first thought?
               - "analysis_process": Step-by-step breakdown of the failure.
               - "final_conclusion": The wrong conclusion reached.
               - "error_reflection": Why it was wrong and how to avoid it.

            OUTPUT JSON ONLY.
            """
        )
        chain = prompt | self.critic_llm
        try:
            res = chain.invoke({
                "case": case_info,
                "answer": final_answer,
                "truth": ground_truth
            })
            content = res.content.strip()
            if content.startswith("```json"): content = content[7:]
            if content.endswith("```"): content = content[:-3]
            return json.loads(content)
        except:
            return {"is_correct": False, "analysis_text": "Parse Error"}