"""Demo script: run the medical agent with PII middleware and show redaction in action."""

import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent import build_agent
from pii_middleware import redact_pii


def run_query(graph, config, query: str, auto_approve: bool = False) -> str:
    """Run a query through the graph, optionally auto-approving sensitive calls."""
    result = graph.invoke({"messages": [HumanMessage(content=query)]}, config)

    state = graph.get_state(config)
    while state.next:
        if auto_approve:
            decision = "approve"
            print("  [auto-approving sensitive tool call for demo]")
        else:
            decision = "reject"
            print("  [auto-rejecting sensitive tool call for demo]")

        result = graph.invoke(Command(resume=decision), config)
        state = graph.get_state(config)

    return result["messages"][-1].content if result["messages"] else ""


def main():
    print("=== PII Middleware Demo ===\n")

    # Show raw vs redacted text
    sample = (
        "Patient John Smith (john.smith@email.com) — "
        "SSN 123-45-6789 — Phone (555) 867-5309"
    )
    print(f"Raw:      {sample}")
    print(f"Redacted: {redact_pii(sample)}\n")

    # Run the agent with PII filter enabled
    print("--- Running agent with PII filter ON ---\n")
    graph = build_agent(pii_filter=True)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    output = run_query(graph, config, "Search for patient John")
    print(f"\nAssistant: {redact_pii(output)}")

    # Run the agent with PII filter disabled for comparison
    print("\n\n--- Running agent with PII filter OFF ---\n")
    graph_raw = build_agent(pii_filter=False)
    config_raw = {"configurable": {"thread_id": str(uuid.uuid4())}}
    output_raw = run_query(graph_raw, config_raw, "Search for patient John")
    print(f"\nAssistant: {output_raw}")


if __name__ == "__main__":
    main()
