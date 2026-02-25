"""Medical assistant agent with human-in-the-loop approval for sensitive operations."""

import json
import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from database import search_patients, get_patient, delete_patient
from human_approval import build_graph
from pii_middleware import pii_redact_tool, redact_pii

load_dotenv()


# --- Tools ---

@tool
def search_patient(query: str) -> str:
    """Search for patients by name or diagnosis. Returns matching patient records."""
    results = search_patients(query)
    if not results:
        return "No patients found matching that query."
    return json.dumps(results, indent=2)


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified address with the given subject and body."""
    print(f"\n{'='*50}")
    print(f"[EMAIL SENT]")
    print(f"  To:      {to}")
    print(f"  Subject: {subject}")
    print(f"  Body:    {body}")
    print(f"{'='*50}\n")
    return f"Email sent successfully to {to}."


@tool
def delete_record(patient_id: str) -> str:
    """Delete a patient record from the database by patient ID."""
    success = delete_patient(patient_id)
    if success:
        print(f"\n[RECORD DELETED] Patient {patient_id} has been removed from the database.\n")
        return f"Patient record {patient_id} has been deleted."
    return f"No patient found with ID {patient_id}."


@tool
def search_medical_literature(query: str) -> str:
    """Search medical literature databases for research papers and clinical guidelines."""
    canned_results = {
        "diabetes": "Recent studies (2024) show GLP-1 receptor agonists reduce cardiovascular risk in T2DM patients. ADA recommends HbA1c target <7% for most adults.",
        "hypertension": "2024 ACC/AHA guidelines recommend BP target <130/80 mmHg. First-line agents: ACE inhibitors, ARBs, CCBs, thiazide diuretics.",
        "asthma": "GINA 2024 update: Low-dose ICS-formoterol as preferred reliever for mild asthma. Step-up therapy based on symptom control.",
        "anxiety": "CBT remains first-line for GAD. SSRIs/SNRIs are first-line pharmacotherapy. Buspirone is an alternative.",
        "migraine": "CGRP monoclonal antibodies (erenumab, fremanezumab) show efficacy for prophylaxis. Acute treatment: triptans, gepants.",
    }
    query_lower = query.lower()
    for key, result in canned_results.items():
        if key in query_lower:
            return result
    return f"Found 3 review articles on '{query}'. Key finding: further research is needed. Consult specialist guidelines for clinical decisions."


# --- Agent setup ---

def build_agent(pii_filter: bool = True):
    """Build and return a LangGraph graph with human-in-the-loop approval."""
    raw_tools = [search_patient, send_email, delete_record, search_medical_literature]

    if pii_filter:
        agent_tools = [pii_redact_tool(t) for t in raw_tools]
    else:
        agent_tools = raw_tools

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    checkpointer = MemorySaver()
    graph = build_graph(agent_tools, llm, checkpointer)
    return graph


# --- REPL ---

def main():
    pii_filter = "--no-pii-filter" not in sys.argv
    graph = build_agent(pii_filter=pii_filter)

    label = "ON" if pii_filter else "OFF"
    print(f"Medical Assistant Agent (PII filter: {label}) — type 'quit' to exit")
    print("Sensitive tools (send_email, delete_record) require human approval.")
    print("-" * 60)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )

        # Check if the graph is interrupted (waiting for approval)
        state = graph.get_state(config)
        while state.next:
            # There's an interrupt — display pending tool calls for approval
            pending = state.tasks
            for task in pending:
                if hasattr(task, "interrupts") and task.interrupts:
                    for intr in task.interrupts:
                        info = intr.value
                        print(f"\n*** APPROVAL REQUIRED ***")
                        print(f"  {info['message']}")
                        for tc in info["tool_calls"]:
                            print(f"  - {tc['name']}({tc['args']})")

            try:
                answer = input("  Approve? (y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return

            decision = "approve" if answer == "y" else "reject"
            result = graph.invoke(Command(resume=decision), config)

            # Check again in case of further interrupts
            state = graph.get_state(config)

        # Extract the final AI message
        final_msg = result["messages"][-1].content if result["messages"] else ""
        output = redact_pii(final_msg) if pii_filter else final_msg
        print(f"\nAssistant: {output}")


if __name__ == "__main__":
    main()
