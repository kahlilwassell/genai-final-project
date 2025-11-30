"""
LangGraph scaffold for the Agentic Run Coach.
Nodes:
- agent: LLM that can call tools
- tools: current tool is a retriever that returns top-k chunks

Helpers:
- run_plan(): multi-week plan generation with profile context + safety review
- run_adjust(): adjust a single day based on weather/fatigue with retrieval grounding
"""

import re
from datetime import date
from typing import List, Sequence, Tuple

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.ingest.retriever import retrieve


load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))


@tool
def retrieve_tool(query: str, k: int = 3, domain: str = "") -> str:
    """
    Retrieve top-k relevant chunks from the training corpus.
    Returns a text block you can cite directly.
    """
    dom = domain if domain else None
    docs = retrieve(query, k=k, domain=dom)
    out_lines: List[str] = []
    for i, d in enumerate(docs, 1):
        out_lines.append(f"[{i}] {d.page_content.strip()}\n(source: {d.metadata.get('source')})")
    return "\n\n".join(out_lines)


def get_model(temperature: float = 0.2) -> ChatOpenAI:
    # Use a stronger model for planning/safety by default
    return ChatOpenAI(model="gpt-4o", temperature=temperature)


def build_graph(temperature: float = 0.2):
    model = get_model(temperature)
    llm_with_tools = model.bind_tools([retrieve_tool])

    sys_msg = SystemMessage(
        content=(
            "You are a running coach. Use the retrieve_tool to ground answers in the training corpus.\n"
            "When calling retrieve_tool, set domain to one of {plans, safety, fueling, biomech} based on the topic.\n"
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


def _rule_based_safety(plan_text: str, profile: str) -> List[str]:
    """
    Heuristic checks: weekly jump, long run cap, back-to-back hard days flag (naive).
    """
    warnings = []
    try:
        current_weekly = float(re.search(r"Current weekly mileage:\s*([\d.]+)", profile).group(1))
    except Exception:
        current_weekly = None
    try:
        recent_lr = float(re.search(r"Recent long run:\s*([\d.]+)", profile).group(1))
    except Exception:
        recent_lr = None

    # Sum distances (very naive: numbers followed by mi)
    miles = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*(?:mi|miles)", plan_text)]
    weekly_total = sum(miles[:10]) if miles else 0  # rough cutoff
    max_long = max(miles) if miles else 0

    if current_weekly:
        cap = current_weekly * 1.15
        if weekly_total and weekly_total > cap:
            warnings.append(f"Weekly volume {weekly_total:.1f} mi exceeds ~15% over current {current_weekly:.1f} mi.")
    if recent_lr and max_long:
        lr_cap = max(recent_lr + 2, recent_lr * 1.2)
        if max_long > lr_cap:
            warnings.append(f"Long run {max_long:.1f} mi is a big jump from recent {recent_lr:.1f} mi.")

    # Back-to-back hard day heuristic
    if "tempo" in plan_text.lower() and "interval" in plan_text.lower():
        if re.search(r"(tempo|interval).{0,40}\n.{0,10}(tempo|interval)", plan_text, re.IGNORECASE):
            warnings.append("Detected possible back-to-back hard sessions (tempo/interval).")

    return warnings


def run_plan(profile: str, task: str, weeks_to_race: int = 12, temperature: float = 0.2) -> Tuple[str, str]:
    """
    Generate a phased plan to race day (weekly summary) plus the next 7-day detailed plan.
    """
    weeks = max(4, min(24, weeks_to_race or 12))
    app = build_graph(temperature=temperature)
    messages = [
        SystemMessage(content="You are a running coach. Follow the system instructions and be safe."),
        HumanMessage(
            content=(
                f"Runner profile: {profile}\n"
                f"Task: {task}\n"
                f"Plan horizon: {weeks} weeks until race.\n"
                "1) Give a week-by-week summary to race day (Base/Build/Taper) with target weekly mileage and key session focus.\n"
                "2) Then give a detailed next-7-day table with Day, Session, Distance, Pace/Effort, and notes. Cite sources like [1].\n"
                "Keep it grounded in the retrieved corpus. If corpus is weak, say so."
            )
        ),
    ]
    result = app.invoke({"messages": messages})
    msgs = result.get("messages", [])
    plan_text = msgs[-1].content if msgs else "No response produced."
    safety_llm = _safety_review(plan_text, profile)
    safety_rules = _rule_based_safety(plan_text, profile)
    combined = safety_llm
    if safety_rules:
        combined = "Heuristic checks:\n- " + "\n- ".join(safety_rules) + "\n\nLLM review:\n" + safety_llm
    return plan_text, combined


def run_adjust(profile: str, today_plan: str, weather: str, fatigue: int, temperature: float = 0.2) -> Tuple[str, str]:
    """
    Adjust a single day based on weather/fatigue. Uses retrieval for guidance.
    """
    app = build_graph(temperature=temperature)
    messages = [
        SystemMessage(
            content=(
                "You are a running coach. Use the retrieve tool to ground adjustments. "
                "Respect safety: reduce intensity/volume for high fatigue or extreme heat. Cite sources."
            )
        ),
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
