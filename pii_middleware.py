"""PII redaction middleware for LangChain tools.

Provides regex-based redaction of emails, phone numbers, and SSNs,
plus a tool wrapper that redacts PII from tool outputs before the LLM sees them.
"""

import re
from functools import wraps
from langchain_core.tools import BaseTool


# --- Regex patterns ---

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(
    r"(\+?1[-.\s]?)?"           # optional country code
    r"(\(?\d{3}\)?[-.\s]?)"     # area code
    r"(\d{3}[-.\s]?\d{4})\b"    # subscriber number
)


def redact_pii(text: str) -> str:
    """Replace emails, SSNs, and phone numbers with redaction placeholders."""
    text = _SSN_RE.sub("[SSN REDACTED]", text)
    text = _EMAIL_RE.sub("[EMAIL REDACTED]", text)
    text = _PHONE_RE.sub("[PHONE REDACTED]", text)
    return text


def pii_redact_tool(original_tool: BaseTool) -> BaseTool:
    """Wrap a LangChain tool so its output is run through redact_pii().

    Returns a new tool with the same name, description, and schema,
    but whose string output has PII stripped.
    """
    original_func = original_tool.func

    @wraps(original_func)
    def wrapped_func(*args, **kwargs):
        result = original_func(*args, **kwargs)
        if isinstance(result, str):
            return redact_pii(result)
        return result

    new_tool = original_tool.model_copy()
    new_tool.func = wrapped_func
    return new_tool
