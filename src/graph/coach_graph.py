"""
LangGraph scaffold for the Agentic Run Coach.
Nodes:
- agent: LLM that can call tools
- tools: current tool is a retriever that returns top-k chunks

Helpers:
- run_plan(): single-turn plan generation with profile context + safety review
- run_adjust(): adjust a single day based on weather/fatigue with retrieval grounding
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
            "You are a running coach. Use the retrieve_tool to ground answers in the training corpus.\n"
            "Return concise, actionable output. Prefer a 7-day table with Day, Session, Distance, and Pace/Effort.\n"
            "Include brief citations like [1] tied to retrieved chunks. If the corpus lacks info, say so."
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


def _safety_review(plan_text: str, profile: str) -> str:
    """
    Light safety reviewer using the base model (no tools).
    Flags: >10% weekly increase, overly long long-run vs profile, hard workouts back-to-back, heat illness warnings.
    """
    llm = get_model(temperature=0.0)
    prompt = (
        "You are a cautious running coach. Review the proposed plan for safety issues.\n"
        "Profile: {profile}\n"
        "Plan:\n{plan}\n\n"
        "List any concrete safety warnings (max 4) or say 'No major safety issues detected.'"
    ).format(profile=profile, plan=plan_text)
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content if hasattr(resp, "content") else str(resp)


def run_plan(profile: str, task: str, temperature: float = 0.2):
    """
    Generate a 7-day plan with citations and a safety review.
    """
    app = build_graph(temperature=temperature)
    messages = [
        SystemMessage(content="You are a running coach. Follow the system instructions and be safe."),
        HumanMessage(
            content=(
                f"Runner profile: {profile}\n"
                f"Task: {task}\n"
                "Produce a 7-day plan table with Day, Session, Distance, Pace/Effort, and brief notes. Cite sources like [1]."
            )
        ),
    ]
    result = app.invoke({"messages": messages})
    msgs = result.get("messages", [])
    plan_text = msgs[-1].content if msgs else "No response produced."
    safety = _safety_review(plan_text, profile)
    return plan_text, safety


def run_adjust(profile: str, today_plan: str, weather: str, fatigue: int, temperature: float = 0.2):
    """
    Adjust a single day based on weather/fatigue. Uses retrieval for guidance.
    """
    app = build_graph(temperature=temperature)
    messages = [
        SystemMessage(content="You are a running coach. Use the retrieve tool to ground adjustments."),
        HumanMessage(
            content=(
                f"Runner profile: {profile}\n"
                f"Today's planned session: {today_plan}\n"
                f"Weather: {weather}\n"
                f"Fatigue (1-5): {fatigue}\n"
                "Adjust the session safely (distance, pace, or modality). Keep it concise and cite sources."
            )
        ),
    ]
    result = app.invoke({"messages": messages})
    msgs = result.get("messages", [])
    adjusted = msgs[-1].content if msgs else "No response produced."
    safety = _safety_review(adjusted, profile)
    return adjusted, safety
