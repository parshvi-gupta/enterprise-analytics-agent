from pathlib import Path
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.analytics_tools import profile_data, grouped_metric, monthly_trend, anomaly_scan
from src.agent import ask_agent

load_dotenv()
st.set_page_config(page_title="Analytics Copilot", page_icon="📊", layout="wide")
st.title("📊 Analytics Copilot")
st.caption("Ask questions about your data. The agent can call a bounded set of analytics tools.")

sample_path = Path("data/sample_sales.csv")
with st.sidebar:
    st.header("Dataset")
    upload = st.file_uploader("Upload a CSV", type=["csv"])
    if upload is not None:
        try:
            df = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()
    else:
        df = pd.read_csv(sample_path)
        st.info("Using the sample sales dataset.")
    st.write(f"**Rows:** {len(df):,}  |  **Columns:** {len(df.columns)}")
    st.write("Columns:", list(df.columns))

st.subheader("Data preview")
st.dataframe(df.head(100), use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Missing cells", int(df.isna().sum().sum()))

with st.expander("Dataset profile"):
    st.json(profile_data(df))

st.subheader("Ask a question")
question = st.text_input("Business question", placeholder="Which region has the highest revenue?")
if st.button("Analyze", type="primary", disabled=not question.strip()):
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Add OPENAI_API_KEY to .env to enable the agent. You can still explore the deterministic analytics below.")
    else:
        with st.spinner("Agent is analyzing and using tools..."):
            try:
                st.markdown(ask_agent(df, question))
            except Exception as exc:
                st.error(f"Agent error: {exc}")

st.subheader("Quick analytics")
with st.expander("Group metric"):
    group_col = st.selectbox("Group by", list(df.columns), key="group")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        metric_col = st.selectbox("Metric", numeric_cols, key="metric")
        operation = st.selectbox("Operation", ["sum", "mean", "median", "count", "min", "max"])
        if st.button("Calculate grouped metric"):
            st.json(grouped_metric(df, group_col, metric_col, operation))
with st.expander("Outlier scan"):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        anomaly_metric = st.selectbox("Column for IQR scan", numeric_cols)
        if st.button("Scan for anomalies"):
            st.json(anomaly_scan(df, anomaly_metric))
