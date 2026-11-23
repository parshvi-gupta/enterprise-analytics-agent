from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.analytics_tools import profile_data, grouped_metric, anomaly_scan
from src.demo_agent import run_demo_analysis
from src.provider_llm import ask_provider

load_dotenv()
st.set_page_config(page_title="Analytics Copilot", page_icon="📊", layout="wide")
st.title("📊 Enterprise Analytics Copilot")
st.caption("Deterministic analytics by default, with optional bring-your-own AI provider.")
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
question = st.text_input("Business question", placeholder="Compare total revenue by region and explain the main pattern")
provider = st.selectbox("AI provider", ["No AI (free demo)", "OpenAI", "Anthropic Claude", "Google Gemini"])
key = ""
model = None
model_options = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o"],
    "Anthropic Claude": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
    "Google Gemini": ["gemini-2.0-flash", "gemini-2.0-pro"],
}
if provider != "No AI (free demo)":
    model = st.selectbox("Model", model_options[provider])
    key = st.text_input(f"{provider} API key", type="password", help="Used in memory for this session; not saved intentionally by the app.")
    st.caption("API keys are separate from consumer ChatGPT/Claude/Gemini subscriptions. API usage may be billed by the provider.")
    st.warning("Privacy: AI mode sends your question and data-derived findings to the selected provider. Findings may contain details derived from the uploaded CSV. Do not upload sensitive or confidential data to this public demo.")
if st.button("Run analysis", type="primary", disabled=not question.strip()):
    with st.spinner("Running specialist analytics..."):
        try:
            result = run_demo_analysis(df, question)
            st.markdown(result["final_answer"])
            if provider != "No AI (free demo)":
                if not key.strip():
                    st.error("Enter your provider API key to use AI, or select No AI.")
                else:
                    with st.spinner(f"Getting interpretation from {provider}..."):
                        try:
                        ai_text = ask_provider(provider, key.strip(), question, result["findings"], model=model)
                        st.subheader(f"{provider} interpretation ({model})")
                        st.markdown(ai_text)
                        payload_chars = len(question) + len(str(result["findings"]))
                        st.caption(f"Rough input-size estimate: ~{max(1, payload_chars // 4):,} tokens. This is not actual token usage or a cost estimate; provider billing varies by model and output.")
                    except Exception as ai_exc:
                        st.error(f"AI provider request failed ({type(ai_exc).__name__}). Check the API key, model availability, provider billing, or quota.")
            with st.expander("Specialist execution trace"):
                st.write("**Selected specialists:**", result.get("specialists", []))
                for event in result.get("audit_log", []):
                    st.write(f"- {event}")
                st.json(result.get("findings", []))
        except Exception as exc:
            st.error(f"Workflow error: {type(exc).__name__}: {exc}")

st.subheader("Deterministic tools (no AI)")
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
st.info("Demo data is synthetic. No AI selected? The app remains fully usable. AI mode sends the question and computed summaries, not the full uploaded dataset.")
