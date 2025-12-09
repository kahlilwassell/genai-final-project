import datetime
from io import StringIO
import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from src.graph.coach_graph import run_plan, run_adjust, run_qa


load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))

st.set_page_config(page_title="Agentic Run Coach", layout="wide")
st.title("Agentic Run Coach")
st.caption("Grounded in your running corpus via FAISS + LangGraph tools.")


def maybe_table(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or "|" not in lines[0]:
        st.markdown(text)
        return
    try:
        cleaned = "\n".join(lines)
        df = pd.read_table(StringIO(cleaned), sep="|")
        df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")
        st.dataframe(df, use_container_width=True)
    except Exception:
        st.markdown(text)


def build_profile_context(data: dict | None) -> dict | str:
    return data or "No profile provided"


def parse_goal_time(time_str: str) -> float | None:
    try:
        parts = time_str.strip().split(":")
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        elif len(parts) == 1:
            h = 0
            m = float(parts[0])
            s = 0
        else:
            return None
        return h * 3600 + m * 60 + s
    except Exception:
        return None


def format_pace(sec_per_unit: float) -> str:
    mins = int(sec_per_unit // 60)
    secs = int(round(sec_per_unit % 60))
    return f"{mins}:{secs:02d}"


def pace_overview(race: str, goal_time_str: str):
    dist_miles = {
        "5K": 3.107,
        "10K": 6.214,
        "Half Marathon": 13.109,
        "Marathon": 26.218,
    }.get(race, None)
    total_sec = parse_goal_time(goal_time_str)
    if dist_miles is None or total_sec is None or total_sec <= 0:
        return "Unable to compute paces from goal time.", {}
    base_pace = total_sec / dist_miles
    paces = {
        "Goal pace": format_pace(base_pace),
        "Easy": format_pace(base_pace + 90),
        "Long run": format_pace(base_pace + 60),
        "Tempo/Threshold": format_pace(base_pace + 20),
        "Interval": format_pace(max(base_pace - 20, base_pace * 0.85)),
    }
    summary = " | ".join(f"{k}: {v}/mi" for k, v in paces.items())
    return summary, paces


tabs = st.tabs(["Setup Profile & Plan", "Ask the Coach", "Adjust a Session"])

# --- Setup Profile & Plan tab ---
with tabs[0]:
    st.subheader("Setup My Profile")
    race_name = st.selectbox("Goal race", ["5K", "10K", "Half Marathon", "Marathon"], index=2)
    race_date = st.date_input("Race date", value=datetime.date.today() + datetime.timedelta(days=90))
    goal_time = st.text_input("Goal race time (hh:mm:ss)", value="01:35:00")
    days_per_week = st.slider("Training days per week", 3, 7, 6)
    weekly_mileage = st.number_input("Current weekly mileage (mi)", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
    long_run = st.number_input("Recent long run (mi)", min_value=0.0, max_value=35.0, value=10.0, step=1.0)
    long_run_day = st.selectbox("Preferred long run day", ["Saturday", "Sunday"], index=1)
    workout_days = st.multiselect(
        "Preferred hard days (tempo/interval)",
        ["Monday","Tuesday", "Wednesday", "Thursday", "Saturday"],
        default=["Tuesday", "Thursday"],
        help="We will spread hard days with easy days between. Long run fixed separately.",
    )

    weeks_to_race = max(4, int((race_date - datetime.date.today()).days // 7))
    pace_summary, pace_dict = pace_overview(race_name, goal_time)
    st.info(f"Pace overview: {pace_summary}")

    if st.button("Generate full plan", type="primary"):
        profile_data = {
            "race_name": race_name,
            "race_date": race_date,
            "goal_time": goal_time,
            "days_per_week": days_per_week,
            "weekly_mileage": weekly_mileage,
            "long_run": long_run,
            "long_run_day": long_run_day,
            "workout_days": workout_days,
        }
        with st.spinner("Generating plan..."):
            plan_text, safety = run_plan(
                profile=profile_data,
                weeks_to_race=weeks_to_race,
                temperature=0.2,
                long_run_day=long_run_day,
                days_per_week=days_per_week,
            )
        st.subheader("Plan")
        maybe_table(plan_text)
        st.subheader("Safety review")
        st.markdown(safety)

# --- Ask the Coach tab ---
with tabs[1]:
    st.subheader("Ask The Coach Question (training, fueling, biomechanics, safety)")
    qa_prompt = st.text_area("Question", value="How should I adjust tempo pace in the heat?", height=120)
    if st.button("Ask", use_container_width=True, key="ask_btn"):
        profile_data = st.session_state.get("profile_data")
        profile = build_profile_context(profile_data)
        with st.spinner("Answering..."):
            answer = run_qa(question=qa_prompt.strip(), profile=profile, temperature=0.2)
        st.subheader("Answer")
        st.markdown(answer)

# --- Adjust tab ---
with tabs[2]:
    st.subheader("Adjust today's session")
    today_plan = st.text_input("Today's planned session", value="6 miles easy + 4x20s strides")
    weather = st.text_input("Weather context", value="85F, humid, sunny")
    fatigue_adjust = st.slider("Fatigue for today (1=fresh, 5=exhausted)", 1, 5, 1, key="fatigue_adjust")
    injury_flag = st.toggle("Injury/illness flag", value=False, key="injury_adjust")
    if st.button("Adjust today", use_container_width=True):
        profile_data = st.session_state.get("profile_data")
        local_profile = dict(profile_data) if profile_data else {}
        local_profile["injury"] = injury_flag
        profile = build_profile_context(local_profile)
        with st.spinner("Adjusting..."):
            adjusted, safety2 = run_adjust(
                profile=profile,
                today_plan=today_plan,
                weather=weather,
                fatigue=fatigue_adjust,
                temperature=0.2,
            )
        st.subheader("Adjusted session")
        st.markdown(adjusted)
        st.subheader("Safety review")
        st.markdown(safety2)
