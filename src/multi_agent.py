"""LangGraph multi-agent analytics workflow.

A bounded supervisor routes requests to fixed specialists. Specialists compute safe,
deterministic evidence from the dataframe before LLM interpretation. A synthesizer
drafts the answer and a critic may trigger one revision.
"""
from __future__ import annotations
import json
import os
from typing import Annotated, Literal, TypedDict
from operator import add
import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from .analytics_tools import profile_data, grouped_metric, monthly_trend, anomaly_scan

SPECIALISTS = ("profiler", "metrics", "trends", "anomalies")

class AgentPlan(BaseModel):
    specialists: list[Literal["profiler", "metrics", "trends", "anomalies"]] = Field(description="Specialists to invoke, in priority order")
    rationale: str = Field(description="Brief routing rationale")

class CriticReview(BaseModel):
    approved: bool = Field(description="Whether the draft is grounded and answers the question")
    feedback: str = Field(description="Specific revision feedback or approval reason")

class AgentState(TypedDict, total=False):
    question: str
    data_summary: dict
    columns: list[str]
    specialists: list[str]
    specialist_cursor: int
    findings: Annotated[list[dict], add]
    audit_log: Annotated[list[str], add]
    draft_answer: str
    final_answer: str
    critique: str
    approved: bool
    revision_attempts: int

def _model() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured. Set it in .env.")
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

def fallback_route(question: str) -> list[str]:
    """Deterministic routing for when structured LLM routing fails."""
    q = question.lower()
    selected: list[str] = []
    if any(w in q for w in ("profile", "missing", "quality", "column", "dataset")):
        selected.append("profiler")
    if any(w in q for w in ("compare", "highest", "lowest", "by region", "by channel", "total", "average", "revenue")):
        selected.append("metrics")
    if any(w in q for w in ("trend", "monthly", "over time", "growth", "season")):
        selected.append("trends")
    if any(w in q for w in ("outlier", "anomaly", "unusual", "spike", "drop")):
        selected.append("anomalies")
    return selected or ["profiler", "metrics"]

def _tool_evidence(df: pd.DataFrame, specialist: str, question: str) -> dict:
    """Compute bounded evidence; the LLM is not asked to calculate from raw rows."""
    if specialist == "profiler":
        return profile_data(df)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    date_candidates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or any(t in str(c).lower() for t in ("date", "time", "month"))]
    categorical = [c for c in df.columns if c not in numeric and c not in date_candidates]
    q = question.lower()
    if specialist == "metrics":
        group = next((c for c in categorical if c.lower() in q), categorical[0] if categorical else None)
        metric = next((c for c in numeric if c.lower() in q), numeric[0] if numeric else None)
        if group and metric:
            return grouped_metric(df, group, metric, "sum")
        return {"categories": categorical, "numeric_metrics": numeric, "note": "No suitable grouping/metric pair could be inferred."}
    if specialist == "trends":
        date_col = date_candidates[0] if date_candidates else None
        metric = next((c for c in numeric if c.lower() in q), numeric[0] if numeric else None)
        if date_col and metric:
            return monthly_trend(df, date_col, metric, "sum")
        return {"date_candidates": date_candidates, "numeric_metrics": numeric, "note": "A date-like column and numeric metric are required."}
    if specialist == "anomalies":
        metric = next((c for c in numeric if c.lower() in q), numeric[0] if numeric else None)
        return anomaly_scan(df, metric) if metric else {"numeric_metrics": numeric, "note": "No numeric metrics available."}
    return {"error": "Unknown specialist"}

def _specialist_node(specialist: str, df: pd.DataFrame):
    def node(state: AgentState) -> dict:
        evidence = _tool_evidence(df, specialist, state["question"])
        prompt = (
            f"You are the {specialist} specialist on an analytics team. Interpret the computed evidence "
            "for the user's question. Do not invent values, make unsupported causal claims, or pretend "
            "you computed anything beyond the evidence. Note limitations.\n\n"
            f"Question: {state['question']}\nEvidence: {json.dumps(evidence, default=str)}"
        )
        try:
            response = _model().invoke(prompt)
            finding = response.content if isinstance(response.content, str) else str(response.content)
        except Exception as exc:
            finding = f"Computed evidence (LLM interpretation unavailable): {json.dumps(evidence, default=str)}. Error type: {type(exc).__name__}"
        return {"findings": [{"specialist": specialist, "evidence": evidence, "finding": finding}],
                "audit_log": [f"{specialist} specialist completed"],
                "specialist_cursor": state.get("specialist_cursor", 0) + 1}
    return node

def _route_node(state: AgentState) -> dict:
    prompt = (
        "You are a supervisor routing a business analytics question. Select the minimum necessary "
        "specialists from: profiler (schema/data quality), metrics (aggregation/comparison), "
        "trends (time patterns), anomalies (outliers). Use their names exactly and in priority order. "
        "If uncertain, use profiler and metrics.\n"
        f"Question: {state['question']}\nColumns: {state.get('columns', [])}\n"
        f"Data profile: {json.dumps(state.get('data_summary', {}), default=str)}"
    )
    try:
        plan = _model().with_structured_output(AgentPlan).invoke(prompt)
        selected = list(dict.fromkeys(plan.specialists))
        rationale = plan.rationale
    except Exception as exc:
        selected = fallback_route(state["question"])
        rationale = f"Deterministic fallback routing ({type(exc).__name__})"
    if not selected:
        selected = fallback_route(state["question"])
    return {"specialists": selected, "specialist_cursor": 0,
            "audit_log": [f"Supervisor routed to {', '.join(selected)}. {rationale}"]}

def _next_specialist(state: AgentState) -> str:
    queue, cursor = state.get("specialists", []), state.get("specialist_cursor", 0)
    return queue[cursor] if cursor < len(queue) else "synthesize"

def _synthesize_node(state: AgentState) -> dict:
    prompt = (
        "You are the lead analytics consultant. Synthesize findings into a clear answer. Ground all "
        "numeric claims in supplied evidence; label hypotheses; state important limitations. "
        "Do not invent additional calculations.\n"
        f"Question: {state['question']}\nFindings: {json.dumps(state.get('findings', []), default=str)}"
    )
    try:
        response = _model().invoke(prompt)
        draft = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        draft = "\n".join(f"- {x['specialist']}: {x['finding']}" for x in state.get("findings", []))
        draft += f"\n\nSynthesis model unavailable ({type(exc).__name__})."
    return {"draft_answer": draft, "audit_log": ["Synthesis agent drafted answer"]}

def _critic_node(state: AgentState) -> dict:
    prompt = (
        "Check if the answer addresses the question and is supported by findings. Look for invented "
        "numbers, causal claims, unsupported conclusions, or missing caveats. Approve if grounded.\n"
        f"Question: {state['question']}\nEvidence: {json.dumps(state.get('findings', []), default=str)}\n"
        f"Draft: {state.get('draft_answer', '')}"
    )
    try:
        review = _model().with_structured_output(CriticReview).invoke(prompt)
        approved, feedback = review.approved, review.feedback
    except Exception as exc:
        approved, feedback = True, f"Critic unavailable ({type(exc).__name__}); review skipped."
    retries = state.get("revision_attempts", 0)
    final = state.get("draft_answer", "")
    if approved or retries >= 1:
        if not approved:
            final += "\n\nReview caveat: " + feedback
        if "Critic unavailable" in feedback:
            final += "\n\nNote: critic review was unavailable."
        return {"approved": approved, "critique": feedback, "final_answer": final,
                "audit_log": [f"Critic finished: approved={approved}"]}
    return {"approved": False, "critique": feedback, "audit_log": [f"Critic requested revision: {feedback}"]}

def _critic_route(state: AgentState) -> str:
    return "revise" if not state.get("approved", False) and state.get("revision_attempts", 0) < 1 else END

def _revise_node(state: AgentState) -> dict:
    prompt = (
        "Revise the draft according to critic feedback. Use only existing evidence, correct unsupported "
        "claims, add caveats, and do not invent calculations.\n"
        f"Question: {state['question']}\nEvidence: {json.dumps(state.get('findings', []), default=str)}\n"
        f"Draft: {state.get('draft_answer', '')}\nFeedback: {state.get('critique', '')}"
    )
    try:
        response = _model().invoke(prompt)
        revised = response.content if isinstance(response.content, str) else str(response.content)
    except Exception:
        revised = state.get("draft_answer", "") + "\n\nRevision note: " + state.get("critique", "Critic feedback")
    return {"draft_answer": revised, "revision_attempts": state.get("revision_attempts", 0) + 1,
            "audit_log": ["Revision agent updated draft"]}

def build_multi_agent_graph(df: pd.DataFrame):
    """Compile the workflow; dataframe is bound to safe analytics specialist nodes."""
    graph = StateGraph(AgentState)
    graph.add_node("router", _route_node)
    for specialist in SPECIALISTS:
        graph.add_node(specialist, _specialist_node(specialist, df))
    graph.add_node("router_dispatch", lambda state: {"audit_log": []})
    graph.add_node("synthesize", _synthesize_node)
    graph.add_node("critic", _critic_node)
    graph.add_node("revise", _revise_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", _next_specialist, {**{name: name for name in SPECIALISTS}, "synthesize": "synthesize"})
    for specialist in SPECIALISTS:
        graph.add_edge(specialist, "router_dispatch")
    graph.add_conditional_edges("router_dispatch", _next_specialist, {**{name: name for name in SPECIALISTS}, "synthesize": "synthesize"})
    graph.add_edge("synthesize", "critic")
    graph.add_conditional_edges("critic", _critic_route, {"revise": "revise", END: END})
    graph.add_edge("revise", "critic")
    return graph.compile()

def run_multi_agent(df: pd.DataFrame, question: str) -> dict:
    """Run the graph and return the answer, selected agents, findings, and trace."""
    numeric = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [str(c) for c in df.columns if c not in numeric and not pd.api.types.is_datetime64_any_dtype(df[c])]
    date_candidates = [str(c) for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or any(t in str(c).lower() for t in ("date", "time", "month"))]
    summary = {"rows": int(len(df)), "columns": list(map(str, df.columns)),
               "numeric_columns": numeric, "categorical_columns": categorical,
               "date_candidates": date_candidates,
               "missing_values": {str(k): int(v) for k, v in df.isna().sum().items()}}
    initial: AgentState = {"question": question, "columns": list(map(str, df.columns)),
        "data_summary": summary, "specialist_cursor": 0, "findings": [],
        "audit_log": [], "revision_attempts": 0}
    return build_multi_agent_graph(df).invoke(initial)
