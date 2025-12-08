"""
LangGraph scaffold for the Agentic Run Coach.
Nodes:
- agent: LLM that can call tools
- tools: retrieval tools (domain-aware) + safety checker

Helpers:
- run_plan(): multi-week plan generation with profile context + safety review
- run_adjust(): adjust a single day based on weather/fatigue with retrieval grounding
"""

import re
from typing import List, Sequence, Tuple

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.ingest.retriever import retrieve


load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))


def _render_docs(docs: List) -> str:
    lines: List[str] = []
    for i, d in enumerate(docs, 1):
        lines.append(f"[{i}] {d.page_content.strip()}\n(source: {d.metadata.get('source')}, domain={d.metadata.get('domain')})")
    return "\n\n".join(lines)


@tool
def retrieve_plans(query: str, k: int = 3) -> str:
    """Retrieve training plan / workout guidance (domain=plans)."""
    return _render_docs(retrieve(query, k=k, domain="plans"))


@tool
def retrieve_safety(query: str, k: int = 3) -> str:
    """Retrieve safety/injury/heat guidance (domain=safety)."""
    return _render_docs(retrieve(query, k=k, domain="safety"))


@tool
def retrieve_fueling(query: str, k: int = 3) -> str:
    """Retrieve fueling/hydration guidance (domain=fueling)."""
    return _render_docs(retrieve(query, k=k, domain="fueling"))


@tool
def retrieve_biomech(query: str, k: int = 3) -> str:
    """Retrieve biomechanics/shoe guidance (domain=biomech)."""
    return _render_docs(retrieve(query, k=k, domain="biomech"))


@tool
def safety_limits(current_weekly: float, recent_long_run: float = 0.0, fatigue: int = 2) -> str:
    """
    Provide basic safety caps based on current mileage, recent long run, and fatigue.
    """
    cap_weekly = current_weekly * 1.1  # ~10% progression
    lr_cap = max(recent_long_run + 2.0, recent_long_run * 1.2)
    fatigue_note = "Reduce intensity/volume by 10–20%" if fatigue >= 4 else "Standard load is acceptable."
    return (
        f"Recommended weekly cap: {cap_weekly:.1f} mi; long run cap: {lr_cap:.1f} mi. "
        f"Fatigue guidance: {fatigue_note}"
    )


@tool
def heat_adjust(temp_f: float, humidity: int = 50) -> str:
    """
    Suggest heat adjustments as % slowdown based on temperature/humidity.
    """
    slow = 0
    if temp_f >= 90:
        slow = 8
    elif temp_f >= 80:
        slow = 5
    elif temp_f >= 75:
        slow = 3
    if humidity >= 70 and slow > 0:
        slow += 1
    note = "Consider swapping to easy/shorter session" if temp_f >= 90 else "Hydrate and adjust pace."
    return f"Suggested slowdown: ~{slow}% at {temp_f}F/{humidity}% humidity. {note}"


def get_model(temperature: float = 0.2) -> ChatOpenAI:
    # Use a stronger model for planning/safety by default
    return ChatOpenAI(model="gpt-4o", temperature=temperature)


def build_graph(temperature: float = 0.2):
    model = get_model(temperature)
    llm_with_tools = model.bind_tools(
        [
            retrieve_plans,
            retrieve_safety,
            retrieve_fueling,
            retrieve_biomech,
            safety_limits,
            heat_adjust,
        ]
    )

    sys_msg = SystemMessage(
        content=(
            "You are a running coach. Use the retrieval tools to ground answers in the training corpus:\n"
            "- retrieve_plans for training plans/structure\n"
            "- retrieve_safety for heat/injury/load guidance\n"
            "- retrieve_fueling for fueling/hydration\n"
            "- retrieve_biomech for shoes/plates/biomechanics\n"
            "Use safety_limits to check volume/long-run caps; use heat_adjust for temperature/humidity adjustments.\n"
            "Return concise, actionable output. Prefer a table with Day, Session, Distance, and Pace/Effort.\n"
            "Include brief citations like [1] tied to retrieved chunks. If the corpus lacks info, say so."
        )
    )

    def agent_node(state: MessagesState):
        msgs: Sequence[BaseMessage] = state["messages"]
        # Ensure the system message is present
        msg_list = list(msgs) if msgs else []
        if not msg_list or not isinstance(msg_list[0], SystemMessage):
            msg_list = [sys_msg] + msg_list
        response = llm_with_tools.invoke(msg_list)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node(
        "tools",
        ToolNode(
            [
                retrieve_plans,
                retrieve_safety,
                retrieve_fueling,
                retrieve_biomech,
                safety_limits,
                heat_adjust,
            ]
        ),
    )
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
        HumanMessage(
            content=(
                f"Runner profile: {profile}\n"
                f"Task: {task}\n"
                f"Plan horizon: {weeks} weeks until race.\n"
                "Use retrieve_tool with domain=plans for training guidance; domain=safety for heat/injury/load caps; domain=fueling for fueling; domain=biomech for footwear/plates.\n"
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
        HumanMessage(
            content=(
                f"Runner profile: {profile}\n"
                f"Today's planned session: {today_plan}\n"
                f"Weather: {weather}\n"
                f"Fatigue (1-5): {fatigue}\n"
                "Use retrieve_tool with domain=safety for heat/fatigue/injury guidance; domain=plans for session structure.\n"
                "Adjust the session safely (distance, pace, or modality). Keep it concise and cite sources."
            )
        ),
    ]
    result = app.invoke({"messages": messages})
    msgs = result.get("messages", [])
    adjusted = msgs[-1].content if msgs else "No response produced."
    safety = _safety_review(adjusted, profile)
    return adjusted, safety
