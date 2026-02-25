# LangChain Guardrails

A demonstration of layered guardrails for a medical assistant agent built with [LangGraph](https://github.com/langchain-ai/langgraph). Each guardrail layer targets a different class of risk, and they compose together into a single graph for comprehensive protection.

## Architecture

```
User Input
    │
    ▼
┌──────────────┐   BLOCK   ┌─────┐
│ Input Guard  │──────────▶│ END │
│ (blocklist + │           └─────┘
│  scope check)│
└──────┬───────┘
       │ PASS
       ▼
┌──────────────┐
│  Agent (LLM) │◀──────────────────────┐
└──────┬───────┘                       │
       │                               │
       ▼                               │
┌──────────────┐   BLOCK   ┌─────┐    │
│ Output Guard │──────────▶│ END │    │
│ (LLM safety  │           └─────┘    │
│  evaluator)  │                       │
└──────┬───────┘                       │
       │ tool calls                    │
       ▼                               │
┌──────────────┐  reject   ┌───────┐  │
│   Approval   │──────────▶│ Agent │──┘
│    Gate      │           └───────┘
└──────┬───────┘
       │ approve
       ▼
┌──────────────┐
│ Tools + PII  │───────────────────────┘
│  Middleware   │
└──────────────┘
```

## Guardrail Layers

| Layer | Type | What it catches |
|---|---|---|
| **Input Guardrail** | Regex blocklist + keyword scope check | Prompt injection (`ignore instructions`, `drop table`, etc.) and off-topic requests |
| **Output Guardrail** | LLM-based safety evaluator (GPT-4o) | Unsafe medical advice, specific dosage recommendations, dangerous procedures |
| **Human Approval Gate** | LangGraph `interrupt()` | Sensitive tool calls (`send_email`, `delete_record`) require explicit approval |
| **PII Middleware** | Regex redaction wrapper on tools | Strips emails, phone numbers, and SSNs from tool outputs before the LLM sees them |

## Files

| File | Purpose |
|---|---|
| `input_guardrail.py` | `InputGuardrail` class with blocklist and medical scope check |
| `output_guardrail.py` | `OutputGuardrail` class with LLM-based safety evaluation |
| `human_approval.py` | Approval gate node, `build_graph()` that wires everything together |
| `pii_middleware.py` | `redact_pii()` function and `pii_redact_tool()` wrapper |
| `agent.py` | Tool definitions (`search_patient`, `send_email`, `delete_record`, `search_medical_literature`) and interactive REPL |
| `database.py` | Fake patient database with embedded prompt injection payloads for testing |
| `layered_guardrails.py` | Automated demo that runs 5 scenarios exercising every guardrail layer |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

## Usage

### Interactive Agent

```bash
python agent.py
```

Starts a REPL where you can chat with the medical assistant. Sensitive tool calls will prompt for approval.

### Layered Guardrails Demo

```bash
python layered_guardrails.py
```

Runs 5 automated test scenarios with colored output showing which guardrail fires at each step:

1. **Prompt injection** — blocked by input guardrail (blocklist match)
2. **Off-topic request** — blocked by input guardrail (scope check)
3. **Unsafe medical advice** — blocked by output guardrail (LLM safety eval)
4. **Sensitive tool call** — human approval gate interrupts, then auto-approves
5. **Normal medical query** — passes all layers successfully
