"""Input guardrail — blocks inappropriate requests before they reach the LLM.

Applies two checks:
1. Keyword blocklist — rejects prompt injection / dangerous patterns
2. Medical scope check — rejects off-topic requests unrelated to medical/patient tasks
"""

import re
from typing import NamedTuple

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, END


class GuardrailResult(NamedTuple):
    allowed: bool
    reason: str


class InputGuardrail:
    BLOCKED_PATTERNS = [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"ignore.*instructions",
            r"disregard.*prompt",
            r"delete\s+all",
            r"drop\s+table",
            r"system\s*:",
            r"admin\s+mode",
            r"override\s+safety",
            r"act\s+as\s+(root|admin)",
            r"reveal.*system\s*prompt",
        ]
    ]

    MEDICAL_KEYWORDS = {
        "patient", "diagnosis", "medication", "prescri", "treatment",
        "doctor", "medical", "health", "symptom", "clinical",
        "record", "email", "search", "literature", "hospital",
        "drug", "therapy", "lab", "test", "nurse", "vital",
        "allergy", "condition", "history", "refer",
    }

    def check(self, user_input: str) -> GuardrailResult:
        # 1. Blocklist check
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.search(user_input):
                return GuardrailResult(
                    allowed=False,
                    reason="Your request was blocked because it matched a restricted pattern. Please rephrase.",
                )

        # 2. Medical scope check
        lower = user_input.lower()
        if any(kw in lower for kw in self.MEDICAL_KEYWORDS):
            return GuardrailResult(allowed=True, reason="")

        return GuardrailResult(
            allowed=False,
            reason="I can only help with medical and patient-related requests. Your message appears to be off-topic.",
        )


def input_guard_node(state: MessagesState) -> dict:
    """Graph node that runs the input guardrail on the latest human message."""
    # Find the last HumanMessage
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            result = InputGuardrail().check(msg.content)
            if not result.allowed:
                return {"messages": [AIMessage(content=result.reason)]}
            return state
    return state


def route_after_guard(state: MessagesState) -> str:
    """Route after the guard node: blocked (AIMessage) → END, otherwise → agent."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        return END
    return "agent"
