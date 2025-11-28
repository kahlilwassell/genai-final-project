import datetime

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.coach_graph import build_graph


load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))

st.set_page_config(page_title="Agentic Run Coach", layout="wide")
st.title("Agentic Run Coach (MVP)")
st.caption("Grounded in your corpus via FAISS + LangGraph tools.")


@st.cache_resource(show_spinner=False)
def get_app(temperature: float):
    return build_graph(temperature=temperature)


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
    value="Create a one-week plan with 1 tempo, 1 long run, and easy days around them.",
    height=120,
)

col1, col2 = st.columns([1, 1])
with col1:
    run_clicked = st.button("Generate", type="primary", use_container_width=True)
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


if run_clicked and prompt.strip():
    app = get_app(temp)
    context_str = build_user_context()
    messages = [
        SystemMessage(content="You are a running coach. Use the provided profile context and retrieval tool. Be safe and actionable."),
        HumanMessage(content=f"Runner profile: {context_str}\n\nQuestion/Task: {prompt.strip()}"),
    ]
    with st.spinner("Generating..."):
        result = app.invoke({"messages": messages})
    msgs = result.get("messages", [])
    answer = msgs[-1].content if msgs else "No response produced."
    st.subheader("Answer")
    st.markdown(answer)
else:
    st.info("Enter a prompt and click Generate.")
