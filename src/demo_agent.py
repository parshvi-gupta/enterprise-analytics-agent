"""Deterministic, no-API demo mode for the analytics copilot."""
from __future__ import annotations

from src.analytics_tools import profile_data, grouped_metric, monthly_trend, anomaly_scan

AGENTS = ("profiler", "metrics", "trends", "anomalies")

def route_question(question: str) -> list[str]:
    q = question.lower()
    agents = []
    if any(k in q for k in ("profile", "missing", "quality", "column", "dataset")):
        agents.append("profiler")
    if any(k in q for k in ("compare", "highest", "lowest", "by ", "total", "average", "revenue", "sales")):
        agents.append("metrics")
    if any(k in q for k in ("trend", "monthly", "over time", "growth", "season")):
        agents.append("trends")
    if any(k in q for k in ("outlier", "anomaly", "unusual", "spike", "drop")):
        agents.append("anomalies")
    return agents or ["profiler", "metrics"]

def run_demo_analysis(df, question: str) -> dict:
    """Run a bounded deterministic specialist workflow without an LLM or API key."""
    specialists = route_question(question)
    findings = []
    numeric = [c for c in df.columns if __import__("pandas").api.types.is_numeric_dtype(df[c])]
    dates = [c for c in df.columns if any(k in str(c).lower() for k in ("date", "time", "month"))]
    categorical = [c for c in df.columns if c not in numeric and c not in dates]
    q = question.lower()
    for agent in specialists:
        if agent == "profiler":
            evidence = profile_data(df)
            text = f"Dataset contains {evidence['rows"]} rows and {len(evidence['columns'])} columns, with {sum(evidence['missing_values'].values())} missing cells."
        elif agent == "metrics":
            group = next((c for c in categorical if c.lower() in q), categorical[0] if categorical else None)
            metric = next((c for c in numeric if c.lower() in q), numeric[0] if numeric else None)
            evidence = grouped_metric(df, group, metric, "sum") if group and metric else {"note":"No suitable grouping and metric were identified.","groups":categorical,"metrics":numeric}
            text = f"Computed a sum aggregation of {metric} by {group}." if group and metric else evidence.get("note","Metrics unavailable.")
        elif agent == "trends":
            date_col = next((c for c in dates if c in df.columns), None)
            metric = next((c for c in numeric if c.lower() in q), numeric[0] if numeric else None)
            evidence = monthly_trend(df, date_col, metric, "sum") if date_col and metric else {"note":"A date-like field and numeric metric are required."}
            text = f"Monthly trend calculated for {metric} using {date_col}." if date_col and metric else evidence["note"]
        else:
            metric = next((c for c in numeric if c.lower() in q), numeric[0] if numeric else None)
            evidence = anomaly_scan(df, metric) if metric else {"note":"No numeric columns found."}
            text = f"IQR anomaly scan performed on {metric}." if metric else evidence["note"]
        findings.append({"specialist": agent, "evidence": evidence, "finding": text})
    answer = "### Demo-mode analysis\n\nThis answer was generated with deterministic Python analytics (no LLM/API call).\n\n"
    answer += "\n".join(f"**{x['specialist'].title()}**: {x['finding']}\n\nEvidence: `{str(x['evidence'])[:1200]}`" for x in findings)
    answer += "\n\n*Limitations: specialists use rule-based routing and fixed analytics; this is not an LLM-generated interpretation.*"
    return {"final_answer": answer, "specialists": specialists, "findings": findings, "audit_log": [f"Deterministic demo routed to: {', '.join(specialists)}"], "approved": True, "revision_attempts": 0}
