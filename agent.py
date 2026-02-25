"""Medical assistant agent with tools for searching patients, sending emails, etc."""

import json
import os
import sys

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

from database import search_patients, get_patient, delete_patient
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
    """Build and return an AgentExecutor, optionally with PII redaction on tools."""
    raw_tools = [search_patient, send_email, delete_record, search_medical_literature]

    if pii_filter:
        agent_tools = [pii_redact_tool(t) for t in raw_tools]
    else:
        agent_tools = raw_tools

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful medical assistant. You help healthcare professionals look up patient information, search medical literature, and manage patient records. Be concise and professional."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_tool_calling_agent(llm, agent_tools, prompt)
    return AgentExecutor(agent=agent, tools=agent_tools, verbose=True)


# --- REPL ---

def main():
    pii_filter = "--no-pii-filter" not in sys.argv
    executor = build_agent(pii_filter=pii_filter)

    label = "ON" if pii_filter else "OFF"
    print(f"Medical Assistant Agent (PII filter: {label}) — type 'quit' to exit")
    print("-" * 55)
    chat_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        result = executor.invoke({"input": user_input, "chat_history": chat_history})
        output = redact_pii(result["output"]) if pii_filter else result["output"]
        print(f"\nAssistant: {output}")


if __name__ == "__main__":
    main()
