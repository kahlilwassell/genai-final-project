import datetime

import pandas as pd
from io import StringIO
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from src.graph.coach_graph import run_plan, run_adjust


load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))

st.set_page_config(page_title="Agentic Run Coach", layout="wide")
st.title("Agentic Run Coach")
st.caption("Grounded in your corpus via FAISS + LangGraph tools.")


with st.sidebar:
    st.subheader("Runner profile")
    race_name = st.text_input("Goal race", value="Half Marathon")
    race_date = st.date_input("Race date", value=datetime.date.today() + datetime.timedelta(days=90))
    weekly_mileage = st.number_input("Current weekly mileage (mi)", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
    long_run = st.number_input("Recent long run (mi)", min_value=0.0, max_value=35.0, value=10.0, step=1.0)
    injury = st.toggle("Injury/illness flag", value=False)
    fatigue = st.slider("Fatigue (1=fresh, 5=exhausted)", 1, 5, 2)

    st.subheader("Model")
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)


prompt = st.text_area(
    "Ask the coach",
    value="Create a plan to race day with weekly targets and a detailed next 7-day block (1 tempo, 1 long run, easy days around them).",
    height=120,
)

col1, col2 = st.columns([1, 1])
with col1:
    run_clicked = st.button("Generate plan", type="primary", use_container_width=True)
with col2:
    st.write("")


def build_user_context() -> str:
    parts = [
        f"Race: {race_name} on {race_date}",
        f"Current weekly mileage: {weekly_mileage} mi",
        f"Recent long run: {long_run} mi",
        f"Injury/illness flag: {injury}",
        f"Fatigue (1-5): {fatigue}",
    ]
    return " | ".join(parts)


def maybe_table(plan_text: str):
    """
    Try to convert a Markdown-like table to DataFrame for nicer display.
    If parsing fails, fallback to raw markdown.
    """
    lines = [ln.strip() for ln in plan_text.splitlines() if ln.strip()]
    if not lines or "|" not in lines[0]:
        st.markdown(plan_text)
        return
    try:
        # Try pandas read_table from markdown-like
        cleaned = "\n".join(lines)
        df = pd.read_table(StringIO(cleaned), sep="|")
        # Drop empty columns that can appear from markdown pipes
        df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")
        st.dataframe(df, use_container_width=True)
    except Exception:
        st.markdown(plan_text)


if run_clicked and prompt.strip():
    profile = build_user_context()
    weeks_to_race = max(4, int((race_date - datetime.date.today()).days // 7))
    with st.spinner("Generating plan..."):
        plan_text, safety = run_plan(
            profile=profile,
            task=prompt.strip(),
            weeks_to_race=weeks_to_race,
            temperature=temp,
        )
    st.subheader("Plan")
    maybe_table(plan_text)
    st.subheader("Safety review")
    st.markdown(safety)
else:
    st.info("Enter a prompt and click Generate plan.")

st.divider()
st.header("Adjust today's session")

today_plan = st.text_input("Today's planned session", value="6 miles easy + 4x20s strides")
weather = st.text_input("Weather context", value="85F, humid, sunny")
fatigue_adjust = st.slider("Fatigue for today (1=fresh, 5=exhausted)", 1, 5, 2, key="fatigue_adjust")

if st.button("Adjust today", use_container_width=True):
    profile = build_user_context()
    with st.spinner("Adjusting..."):
        adjusted, safety2 = run_adjust(
            profile=profile,
            today_plan=today_plan,
            weather=weather,
            fatigue=fatigue_adjust,
            temperature=temp,
        )
    st.subheader("Adjusted session")
    st.markdown(adjusted)
    st.subheader("Safety review")
    st.markdown(safety2)
