from pathlib import Path
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.analytics_tools import profile_data, grouped_metric, anomaly_scan
from src.demo_agent import run_demo_analysis
from src.multi_agent import run_multi_agent

load_dotenv()
st.set_page_config(page_title="Analytics Copilot | Multi-Agent", page_icon="📊", layout="wide")
st.title("📊 Analytics Copilot")
st.caption("Bounded analytics specialists with deterministic demo mode and optional LLM mode.")
sample_path = Path("data/sample_sales.csv")

with st.sidebar:
    st.header("Dataset")
    upload = st.file_uploader("Upload CSV", type=["csv"])
    if upload is not None:
        try:
            df = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Unable to parse CSV: {exc}")
            st.stop()
    else:
        df = pd.read_csv(sample_path)
        st.info("Sample sales dataset loaded.")
    st.write(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
    st.write("Fields:", list(df.columns))

st.subheader("Data overview")
k1, k2, k3 = st.columns(3)
k1.metric("Records", f"{len(df):,}")
k2.metric("Columns", len(df.columns))
k3.metric("Missing values", int(df.isna().sum().sum()))
st.dataframe(df.head(50), use_container_width=True)
with st.expander("Profile data"):
    st.json(profile_data(df))

st.subheader("Ask the analytics team")
question = st.text_input("Question", placeholder="Compare total revenue by region and explain the main pattern")
api_available = bool(os.getenv("OPENAI_API_KEY"))
use_llm = st.checkbox("Use optional LLM-powered workflow (requires API key)", value=False, disabled=not api_available, help="Demo mode works without credentials. LLM mode requires your own API key.")
if not api_available:
    st.info("No API key detected — deterministic demo mode is enabled. All core analytics and specialist routing work without credentials.")
if st.button("Run analysis", type="primary", disabled=not question.strip()):
    with st.spinner("Running specialist analytics..."):
        try:
            result = run_multi_agent(df, question) if use_llm and api_available else run_demo_analysis(df, question)
            st.markdown(result.get("final_answer", result.get("draft_answer", "No answer was produced.")))
            with st.expander("Agent execution trace"):
                st.write("**Selected specialists:**", result.get("specialists", []))
                st.write("**Workflow log:**")
                for event in result.get("audit_log", []):
                    st.write(f"- {event}")
                st.write("**Evidence and specialist findings:**")
                st.json(result.get("findings", []))
                st.write("**Critic approved:**", result.get("approved"))
                st.write("**Revision attempts:**", result.get("revision_attempts", 0))
        except Exception as exc:
            st.error(f"Workflow error: {type(exc).__name__}: {exc}")

st.subheader("Deterministic tools (no LLM)")
with st.expander("Group a metric"):
    categorical = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if categorical and numeric:
        group_col = st.selectbox("Group by", categorical, key="group")
        metric_col = st.selectbox("Metric", numeric, key="metric")
        op = st.selectbox("Operation", ["sum", "mean", "median", "count", "min", "max"])
        if st.button("Calculate", key="calculate_grouped"):
            st.json(grouped_metric(df, group_col, metric_col, op))
with st.expander("IQR outlier scan"):
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric:
        anomaly_metric = st.selectbox("Metric to scan", numeric)
        if st.button("Scan", key="scan_outliers"):
            st.json(anomaly_scan(df, anomaly_metric))

st.info("Portfolio demo uses synthetic sample data. Analytics are deterministic; no arbitrary code execution or external data writes.")
