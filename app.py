from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.analytics_tools import profile_data, grouped_metric, monthly_trend, anomaly_scan
from src.demo_agent import run_demo_analysis
from src.provider_llm import ask_provider

load_dotenv()
st.set_page_config(page_title="Enterprise Analytics Copilot", page_icon="📊", layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(180deg,#f5f8fc 0%,#fff 40%);color:#10243a}
.block-container{max-width:1320px;padding-top:1rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:#f0f4f9;border-right:1px solid #e1e8f0}
[data-testid="stMetric"]{background:#fff;border:1px solid #e3ebf3;border-radius:16px;padding:16px 18px;box-shadow:0 3px 14px rgba(16,36,58,.04)}
[data-testid="stMetricLabel"] p{color:#6b7d90;font-size:.78rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase}
[data-testid="stMetricValue"]{color:#10243a;font-weight:750}
.stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid #e4ebf2}
.stTabs [data-baseweb="tab"]{height:48px;border-radius:12px 12px 0 0;padding:0 16px;font-weight:650}
.stTabs [aria-selected="true"]{color:#168e83!important;border-bottom-color:#19a89b!important}
div[data-testid="stDataFrame"],div[data-testid="stTable"]{border:1px solid #e4ebf2;border-radius:14px;overflow:hidden}
div[data-testid="stPlotlyChart"]{background:white;border:1px solid #e8eef4;border-radius:16px;padding:8px}
button[kind="primary"]{background:linear-gradient(90deg,#147e78,#19a89b);border:0;border-radius:10px;font-weight:700}
.hero{padding:24px 26px;border-radius:22px;background:linear-gradient(120deg,#10263d,#183c56 62%,#147e78);color:white;margin:0 0 20px;box-shadow:0 10px 30px rgba(16,36,58,.12)}
.hero h1{font-size:2.1rem;margin:0 0 7px;letter-spacing:-.04em;color:white}
.hero p{color:#d1e5ed;margin:0;font-size:1rem}
.hero .eyebrow{font-size:.74rem;font-weight:800;letter-spacing:.14em;color:#83ded3;margin-bottom:8px}
.section-label{color:#718399;font-size:.76rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}
</style>""", unsafe_allow_html=True)
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
[data-testid="stMetric"] {background:rgba(120,120,120,.07); padding:14px 16px; border-radius:14px; border:1px solid rgba(120,120,120,.15);}
div[data-testid="stTabs"] button {font-weight:600;}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="hero"><div class="eyebrow">AI-ASSISTED BUSINESS ANALYTICS</div><h1>Enterprise Analytics Copilot</h1><p>From raw CSV to business-ready findings — explore, measure, investigate and ask questions in one workspace.</p></div>', unsafe_allow_html=True)

@st.cache_data
def load_sample():
    return pd.read_csv(Path("data/sample_sales.csv"))

with st.sidebar:
    st.header("Workspace")
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            st.stop()
    else:
        df = load_sample()
        st.info("Using sample sales data")
    st.divider()
    st.caption("Current dataset")
    st.write(f"**{len(df):,} rows** · **{len(df.columns)} columns**")
    st.caption("Fields")
    st.write(", ".join(map(str, df.columns)))

if df.empty:
    st.warning("This dataset is empty. Upload a CSV with at least one row to continue.")
    st.stop()

numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
categorical = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
date_cols = [c for c in df.columns if any(k in str(c).lower() for k in ("date", "time", "month"))]

m1,m2,m3,m4 = st.columns(4)
m1.metric("Records", f"{len(df):,}")
m2.metric("Columns", f"{len(df.columns):,}")
m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
m4.metric("Numeric fields", f"{len(numeric):,}")

# Data health insights
st.markdown("### Dataset health signals")
h1,h2,h3 = st.columns(3)
missing_pct = (df.isna().sum().sum() / max(1, df.size)) * 100
h1.metric("Completeness", f"{100-missing_pct:.1f}%")
h2.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
h3.metric("Largest numeric spread", max(numeric, key=lambda col: float(df[col].std(skipna=True) or 0)) if numeric else "—")
with st.expander("Download data profile"):
    profile_frame = pd.DataFrame({"column":df.columns.astype(str),"dtype":df.dtypes.astype(str),"missing":df.isna().sum().values,"missing_pct":(df.isna().mean()*100).round(1).values,"unique":df.nunique(dropna=True).values})
    st.download_button("Download profile CSV",profile_frame.to_csv(index=False).encode("utf-8"),"dataset_profile.csv","text/csv")

overview, explore, ask, tools = st.tabs(["Overview", "Explore", "AI Copilot", "Analytics tools"])
with overview:
    left,right = st.columns([1.25,1])
    with left:
        st.subheader("Data preview")
        st.dataframe(df.head(100), use_container_width=True, height=350)
    with right:
        st.subheader("Column types")
        type_df = pd.DataFrame({"Field":df.columns.astype(str), "Type":df.dtypes.astype(str), "Missing":df.isna().sum().values, "Unique":df.nunique(dropna=True).values})
        st.dataframe(type_df, hide_index=True, use_container_width=True, height=350)
    with st.expander("Detailed profile"):
        st.json(profile_data(df))
with explore:
    st.subheader("Explore your data")
    if categorical and numeric:
        c1,c2,c3 = st.columns(3)
        group_col = c1.selectbox("Break down by", categorical, key="explore_group")
        metric_col = c2.selectbox("Measure", numeric, key="explore_metric")
        operation = c3.selectbox("Aggregation", ["sum","mean","median","count","min","max"], key="explore_op")
        result = grouped_metric(df, group_col, metric_col, operation)
        if "results" in result:
            result_df = pd.DataFrame(list(result["results"].items()), columns=[group_col, f"{operation} of {metric_col}"])
            st.bar_chart(result_df.set_index(group_col))
            st.dataframe(result_df, hide_index=True, use_container_width=True)
        else:
            st.warning(result.get("error", "Unable to calculate this metric."))
    else:
        st.info("Exploration charts need at least one categorical and one numeric field.")
    if date_cols and numeric:
        st.divider()
        c1,c2 = st.columns(2)
        date_col = c1.selectbox("Date field", date_cols, key="trend_date")
        trend_metric = c2.selectbox("Trend metric", numeric, key="trend_metric")
        trend = monthly_trend(df, date_col, trend_metric, "sum")
        if "trend" in trend and trend["trend"]:
            trend_df = pd.DataFrame(list(trend["trend"].items()), columns=["Month","Value"]).sort_values("Month")
            st.line_chart(trend_df.set_index("Month"))
        else:
            st.caption("No valid dates were found in the selected column.")
with ask:
    st.subheader("Ask a business question")
    question = st.text_area("Your question", placeholder="Compare revenue by region. Which region is strongest and what patterns stand out?", height=90)
    provider = st.selectbox("Interpretation mode", ["No AI — free demo", "OpenAI", "Anthropic Claude", "Google Gemini"])
    model_options = {
        "OpenAI": ["gpt-4o-mini", "gpt-4o"],
        "Anthropic Claude": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
        "Google Gemini": ["gemini-2.0-flash", "gemini-2.0-pro"],
    }
    key, model = "", None
    if provider != "No AI — free demo":
        model = st.selectbox("Model", model_options[provider])
        key = st.text_input("API key", type="password", help="Used for this request; the app does not intentionally save it.")
        st.caption("Provider API access may be billed separately from consumer subscriptions.")
        st.warning("Privacy: only your question and computed findings are sent to the provider. Findings may contain derived information. Avoid confidential data in this public demo.")
    run = st.button("Run analysis", type="primary", disabled=not question.strip())
    if run:
        with st.spinner("Running analytics specialists..."):
            try:
                result = run_demo_analysis(df, question)
                st.markdown(result["final_answer"])
                if provider != "No AI — free demo":
                    if not key.strip():
                        st.error("Enter an API key, or select free demo mode.")
                    else:
                        with st.spinner("Generating evidence-based interpretation..."):
                            try:
                                ai_text = ask_provider(provider, key.strip(), question, result["findings"], model=model)
                                st.markdown("#### AI interpretation")
                                st.markdown(ai_text)
                                payload_chars = len(question) + len(str(result["findings"]))
                                st.caption(f"Approximate input-size indicator: ~{max(1, payload_chars // 4):,} tokens (not actual usage or cost).")
                            except Exception as exc:
                                st.error(f"Provider request failed ({type(exc).__name__}). Check API key, model, billing, or quota.")
                with st.expander("Specialist trace and evidence"):
                    st.write("**Selected specialists:**", result.get("specialists", []))
                    for event in result.get("audit_log", []):
                        st.write(f"• {event}")
                    st.json(result.get("findings", []))
            except Exception as exc:
                st.error(f"Analysis failed ({type(exc).__name__}): {exc}")
    st.info("Free demo mode runs deterministic Python analytics without an API key. AI mode interprets computed evidence; it does not directly inspect the raw CSV.")
with tools:
    st.subheader("Toolbox")
    if numeric:
        metric = st.selectbox("Numeric field to scan", numeric, key="outlier_metric")
        if st.button("Find unusual values"):
            st.json(anomaly_scan(df, metric))
    else:
        st.info("No numeric fields available for anomaly detection.")
    st.caption("Anomaly scans use the interquartile range (IQR) rule and are intended for exploration, not automatic decisions.")
