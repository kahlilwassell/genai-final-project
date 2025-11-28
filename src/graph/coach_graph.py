"""
LangGraph scaffold for the Agentic Run Coach.
Nodes:
- agent: LLM that can call tools
- tools: current tool is a retriever that returns top-k chunks

This is an early skeleton; planner/adjuster/safety logic can be added in separate nodes later.
"""

from typing import List, Sequence

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.ingest.retriever import retrieve


load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))


@tool
def retrieve_tool(query: str, k: int = 3) -> str:
    """
    Retrieve top-k relevant chunks from the training corpus.
    Returns a text block you can cite directly.
    """
    docs = retrieve(query, k=k)
    out_lines: List[str] = []
    for i, d in enumerate(docs, 1):
        out_lines.append(f"[{i}] {d.page_content.strip()}\n(source: {d.metadata.get('source')})")
    return "\n\n".join(out_lines)


def get_model(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


def build_graph(temperature: float = 0.2):
    model = get_model(temperature)
    llm_with_tools = model.bind_tools([retrieve_tool])

    sys_msg = SystemMessage(
        content=(
            "You are a running coach. Use the retrieve_tool to ground answers in the training corpus. "
            "Be concise and actionable. Include brief citations like [1] or (source). "
            "If the corpus does not cover the question, say so."
        )
    )

    def agent_node(state: MessagesState):
        msgs: Sequence[BaseMessage] = state["messages"]
        if not msgs:
            return {"messages": []}
        response = llm_with_tools.invoke(list(msgs))
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode([retrieve_tool]))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_coach(user_query: str, temperature: float = 0.2):
    """
    Convenience helper to run a single-turn query through the graph.
    """
    app = build_graph(temperature=temperature)
    state: MessagesState = {"messages": [HumanMessage(content=user_query)]}
    result = app.invoke(state)
    msgs = result.get("messages", [])
    return msgs[-1].content if msgs else ""
