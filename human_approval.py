"""Human-in-the-loop approval gate for sensitive tool calls.

Uses LangGraph's interrupt() to pause execution when the agent tries to call
sensitive tools (send_email, delete_record), requiring explicit human approval
before proceeding.
"""

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command

from input_guardrail import input_guard_node, route_after_guard
from output_guardrail import build_output_guard_node, route_after_output_guard


SENSITIVE_TOOLS = {"send_email", "delete_record"}


def agent_node(state: MessagesState, llm_with_tools: Any) -> dict:
    """Invoke the LLM and return its response."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def approval_check(state: MessagesState) -> Command:
    """Gate node: interrupt for approval if the last message contains sensitive tool calls."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return Command(goto="tools")

    sensitive_calls = [tc for tc in last.tool_calls if tc["name"] in SENSITIVE_TOOLS]

    if not sensitive_calls:
        # All calls are safe — proceed directly to tool execution
        return Command(goto="tools")

    # Interrupt and present the sensitive calls for human review
    decision = interrupt({
        "message": "Sensitive tool call(s) detected. Approve?",
        "tool_calls": [
            {"id": tc["id"], "name": tc["name"], "args": tc["args"]}
            for tc in sensitive_calls
        ],
    })

    if decision == "approve":
        return Command(goto="tools")

    # Rejected — return ToolMessages so the LLM knows the calls were denied
    rejection_msgs = []
    for tc in last.tool_calls:
        if tc["name"] in SENSITIVE_TOOLS:
            rejection_msgs.append(
                ToolMessage(
                    content=f"Tool call '{tc['name']}' was rejected by the user.",
                    tool_call_id=tc["id"],
                )
            )
        else:
            # Let safe calls that were bundled with sensitive ones also get rejected
            # so the ToolNode doesn't receive a partial set.
            rejection_msgs.append(
                ToolMessage(
                    content=f"Tool call '{tc['name']}' was skipped because the batch was rejected.",
                    tool_call_id=tc["id"],
                )
            )
    return Command(goto="agent", update={"messages": rejection_msgs})


def build_graph(tools, llm, checkpointer):
    """Build and compile the LangGraph StateGraph with an approval gate.

    Args:
        tools: list of LangChain tools to bind to the LLM
        llm: ChatOpenAI (or compatible) instance
        checkpointer: a LangGraph checkpointer (e.g. MemorySaver)

    Returns:
        Compiled LangGraph graph
    """
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(MessagesState)

    # Nodes
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("agent", lambda state: agent_node(state, llm_with_tools))
    graph.add_node("output_guard", build_output_guard_node(llm))
    graph.add_node("approval_check", approval_check)
    graph.add_node("tools", ToolNode(tools))

    # Edges
    graph.set_entry_point("input_guard")
    graph.add_conditional_edges("input_guard", route_after_guard, {"agent": "agent", END: END})
    graph.add_edge("agent", "output_guard")
    graph.add_conditional_edges("output_guard", route_after_output_guard, {"approval_check": "approval_check", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
