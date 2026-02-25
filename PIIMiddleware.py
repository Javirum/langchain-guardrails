"""Demo script: run the medical agent with PII middleware and show redaction in action."""

from agent import build_agent
from pii_middleware import redact_pii


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
    executor = build_agent(pii_filter=True)
    result = executor.invoke({
        "input": "Search for patient John",
        "chat_history": [],
    })
    output = redact_pii(result["output"])
    print(f"\nAssistant: {output}")

    # Run the agent with PII filter disabled for comparison
    print("\n\n--- Running agent with PII filter OFF ---\n")
    executor_raw = build_agent(pii_filter=False)
    result_raw = executor_raw.invoke({
        "input": "Search for patient John",
        "chat_history": [],
    })
    print(f"\nAssistant: {result_raw['output']}")


if __name__ == "__main__":
    main()
