"""Output guardrail — LLM-based safety validation of agent responses.

Evaluates the agent's final text responses for safety issues such as harmful
medical advice, hallucinated treatments, self-harm instructions, or unauthorized
disclosures. Blocks unsafe responses with a refusal message.
"""

import json
from typing import NamedTuple

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import MessagesState, END


class GuardrailResult(NamedTuple):
    allowed: bool
    reason: str


SAFETY_PROMPT = """You are a safety evaluator for a medical assistant chatbot. Your job is to determine whether the assistant's response is safe to show to the user.

Flag the response as UNSAFE if it contains any of the following:
- Specific dosage recommendations or prescription advice without citing a verified source
- Encouragement to self-harm, self-medicate, or avoid seeking professional care
- Hallucinated drug names, fabricated clinical trials, or invented treatment protocols
- Unauthorized disclosure of other patients' information
- Instructions for obtaining controlled substances illegally
- Dangerous medical procedures the user could attempt at home
- Definitive diagnoses that should only come from a qualified clinician

Flag the response as SAFE if it:
- Provides general health education with appropriate disclaimers
- Recommends consulting a healthcare professional
- Shares publicly available medical guidelines with proper attribution
- Performs administrative tasks (searching records, sending emails) without unsafe medical content

Respond with ONLY a JSON object in this exact format, no other text:
{"safe": true, "reason": "brief explanation"}
or
{"safe": false, "reason": "brief explanation of the safety concern"}"""


class OutputGuardrail:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, response_text: str) -> GuardrailResult:
        result = self.llm.invoke([
            SystemMessage(content=SAFETY_PROMPT),
            HumanMessage(content=f"Evaluate this assistant response:\n\n{response_text}"),
        ])
        try:
            parsed = json.loads(result.content)
            return GuardrailResult(
                allowed=parsed["safe"],
                reason=parsed.get("reason", ""),
            )
        except (json.JSONDecodeError, KeyError):
            # If we can't parse the safety check, block out of caution
            return GuardrailResult(
                allowed=False,
                reason="Safety check produced an unparseable response; blocked as a precaution.",
            )


REFUSAL_MESSAGE = (
    "I'm sorry, but I can't provide that response as it was flagged by our "
    "safety review. Please consult a qualified healthcare professional for "
    "medical advice."
)


def build_output_guard_node(llm):
    """Return an output_guard node closure with the LLM captured."""
    guardrail = OutputGuardrail(llm)

    def output_guard_node(state: MessagesState) -> dict:
        last = state["messages"][-1]

        # Only evaluate final text responses; pass tool-call messages through
        if not isinstance(last, AIMessage) or last.tool_calls:
            return state

        result = guardrail.evaluate(last.content)
        if result.allowed:
            return state

        # Replace the unsafe response with a refusal
        return {"messages": [AIMessage(content=REFUSAL_MESSAGE)]}

    return output_guard_node


def route_after_output_guard(state: MessagesState) -> str:
    """Route after output guard: tool calls → approval_check, otherwise → END."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "approval_check"
    return END
