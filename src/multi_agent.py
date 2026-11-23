"""LangGraph multi-agent analytics workflow.

The workflow is intentionally bounded: a router selects from four fixed specialists,
specialists produce evidence-backed findings, a synthesizer drafts the answer, and
a critic may trigger one revision. No agent can execute arbitrary Python or SQL.
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
    specialists: list[Literal["profiler", "metrics", "trends", "anomalies"]] = Field(
        description="Specialists to invoke in priority order. Choose only those needed."
    )
    rationale: str = Field(description="Brief routing rationale")


class CriticReview(BaseModel):
    approved: bool = Field(description="Whether the answer is grounded and sufficiently complete")
    feedback: str = Field(description="Specific issue to fix, or approval reason")


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


def _model():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured. Set it in .env.")
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


def fallback_route(question: str) -> list[str]:
    """Deterministic route used when the LLM router is unavailable."""
    q = question.lower()
    selected: list[str] = []
    if any(word in q for word in ("profile", "missing", "quality", "column", "dataset")):
        selected.append("profiler")
    if any(word in q for word in ("compare", "highest", "lowest", "by region", "by channel", "total", "average", "revenue")):
        selected.append("metrics")
    if any(word in q for word in ("trend", "monthly", "over time", "growth", "season")):
        selected.append("trends")
    if any(word in q for word in ("outlier", "anomaly", "unusual", "spike", "drop")):
        selected.append("anomalies")
    return selected or ["profiler", "metrics"]


def _json_safe(value):
    return json.loads(json.dumps(value, default=str))


def _tool_evidence(df: pd.DataFrame, specialist: str, question: str) -> dict:
    """Generate deterministic, bounded evidence before any LLM interpretation."""
    if specialist == "profiler":
        return profile_data(df)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not dates:
        dates = [c for c in df.columns if any(token in c.lower() for token in ("date", "time", "month"))]
    categorical = [c for c in df.columns if c not in numeric and c not in dates]
    q = question.lower()
    if specialist == "metrics":
        group = next((c for c in categorical if c.lower() in q), None)
        if group is None:
            group = next((c for c in categorical if c.lower() in q.replace("by ", "")), categorical[0] if categorical else None)
        metric = next((c for c in numeric if c.lower() in q), numeric[0] if numeric else None)
        if group and metric:
            return grouped_metric(df, group, metric, "sum")
        return {"available_categories": categorical, "available_numeric_metrics": numeric, "note": "No obvious grouping or metric found; list available fields."}
    if specialist == "trends":
        date_col = dates[0] if dates else None
        metric = next((c for c in numeric if c.lower() in q), numeric[0] if numeric else None)
        if date_col and metric:
            return monthly_trend(df, date_col, metric, "sum")
        return {"date_candidates": dates, "numeric_metrics": numeric, "note": "Trend requires a date-like and numeric column."}
    if specialist == "anomalies":
        metric = next((c for c in numeric if c.lower() in q), numeric[0] if numeric else None)
        if metric:
            return anomaly_scan(df, metric)
        return {"numeric_metrics": numeric, "note": "No numeric columns available."}
    return {"error": "Unknown specialist"}


def _specialist_node(specialist: str):
    def node(state: AgentState) -> dict:
        evidence = _tool_evidence_from_summary(state, specialist)
        # The graph receives only bounded, deterministic evidence—not raw dataframe rows.
        prompt = (
            f"You are the {specialist} specialist in a business analytics team. "
            "Interpret the supplied computed evidence to answer the business question. "
            "Do not invent values, claim causal relationships, or suggest unsupported precision. "
            "If evidence is insufficient, say what is missing.\n\n"
            f"Question: {state['question']}\nEvidence: {json.dumps(evidence, default=str)}"
        )
        try:
            result = _model().invoke(prompt)
            finding = result.content if isinstance(result.content, str) else str(result.content)
        except Exception as exc:
            finding = f"Evidence summary (LLM interpretation unavailable): {json.dumps(evidence, default=str)}. Model error: {type(exc).__name__}"
        return {
            "findings": [{"specialist": specialist, "evidence": evidence, "finding": finding}],
            "audit_log": [f"{specialist} specialist completed"],
            "specialist_cursor": state.get("specialist_cursor", 0) + 1,
        }
    return node


def _tool_evidence_from_summary(state: AgentState, specialist: str) -> dict:
    """Use an already-computed safe profile, or fallback schema-only evidence."""
    # The graph stores an evidence-safe profile, not the entire uploaded dataframe.
    profile = state.get("data_summary", {})
    if specialist == "profiler":
        return profile
    columns = state.get("columns", [])
    numeric = profile.get("numeric_columns", [])
    categorical = profile.get("categorical_columns", [])
    question = state.get("question", "")
    if specialist == "metrics":
        return {"available_categories": categorical, "available_numeric_metrics": numeric, "computed_profile": profile}
    if specialist == "trends":
        return {"date_candidates": profile.get("date_candidates", []), "numeric_metrics": numeric, "computed_profile": profile}
    if specialist == "anomalies":
        return {"numeric_metrics": numeric, "computed_profile": profile}
    return {"columns": columns}


def _profile_node(state: AgentState) -> dict:
    return {"audit_log": ["Router evaluated request"], "specialist_cursor": 0}


def _route_node(state: AgentState) -> dict:
    columns = state.get("columns", [])
    question = state["question"]
    schema = state.get("data_summary", {})
    prompt = (
        "Route a business analytics question to the minimum relevant specialists. "
        "Available specialists: profiler (schema/missingness/quality), metrics (aggregations/comparisons), "
        "trends (time patterns), anomalies (outliers). Select only specialists needed, in logical order. "
        "If unsure, choose profiler and metrics.\n"
        f"Question: {question}\nColumns: {columns}\nProfile: {json.dumps(schema, default=str)}"
    )
    try:
        plan = _model().with_structured_output(AgentPlan).invoke(prompt)
        selected = list(dict.fromkeys(plan.specialists))
        rationale = plan.rationale
    except Exception as exc:
        selected = fallback_route(question)
        rationale = f"Rule-based fallback router used ({type(exc).__name__})."
    if not selected:
        selected = fallback_route(question)
    return {
        "specialists": selected,
        "specialist_cursor": 0,
        "audit_log": [f"Router selected: {', '.join(selected)}. {rationale}"],
    }


def _next_specialist(state: AgentState) -> str:
    queue = state.get("specialists", [])
    cursor = state.get("specialist_cursor", 0)
    if cursor >= len(queue):
        return "synthesize"
    return queue[cursor]


def _synthesize_node(state: AgentState) -> dict:
    prompt = (
        "You are the lead analytics consultant. Synthesize the specialist findings into a concise, "
        "useful business answer. Ground every numeric statement in the supplied evidence. Clearly label "
        "hypotheses and limitations. Do not invent additional calculations.\n"
        f"Question: {state['question']}\nSpecialist findings: {json.dumps(state.get('findings', []), default=str)}"
    )
    try:
        draft = _model().invoke(prompt).content
        if not isinstance(draft, str):
            draft = str(draft)
    except Exception as exc:
        draft = "Specialist evidence:\n" + "\n".join(f"- {x.get('specialist')}: {x.get('finding')}" for x in state.get("findings", [])) + f"\n(Synthesis model unavailable: {type(exc).__name__})"
    return {"draft_answer": draft, "audit_log": ["Synthesis agent drafted response"]}


def _critic_node(state: AgentState) -> dict:
    prompt = (
        "Review this analytics answer for unsupported claims, mismatched metrics, missing caveats, "
        "or failure to answer the question. Only approve if it is grounded in the findings. "
        "Do not demand stylistic perfection.\n"
        f"Question: {state['question']}\nFindings: {json.dumps(state.get('findings', []), default=str)}\n"
        f"Draft: {state.get('draft_answer', '')}"
    )
    try:
        review = _model().with_structured_output(CriticReview).invoke(prompt)
        approved, feedback = review.approved, review.feedback
    except Exception as exc:
        # If critic service fails, don't block a bounded answer. Surface the limitation.
        approved, feedback = True, f"Critic unavailable ({type(exc).__name__}); review skipped."
    retries = state.get("revision_attempts", 0)
    final = state.get("draft_answer", "")
    if approved or retries >= 1:
        if not approved:
            final += "\n\nReview caveat: " + feedback
        if "Critic unavailable" in feedback:
            final += "\n\nNote: automated critic review was unavailable for this run."
        return {"approved": approved, "critique": feedback, "final_answer": final, "audit_log": [f"Critic finished: approved={approved}"]}
    return {"approved": False, "critique": feedback, "audit_log": [f"Critic requested revision: {feedback}"]}


def _critic_route(state: AgentState) -> str:
    if not state.get("approved", False) and state.get("revision_attempts", 0) < 1:
        return "revise"
    return END


def _revise_node(state: AgentState) -> dict:
    prompt = (
        "Revise this analytics answer using the critic feedback. Preserve only evidence-backed claims, "
        "add caveats if needed, and do not create new calculations.\n"
        f"Question: {state['question']}\nEvidence: {json.dumps(state.get('findings', []), default=str)}\n"
        f"Draft: {state.get('draft_answer', '')}\nCritic feedback: {state.get('critique', '')}"
    )
    try:
        revised = _model().invoke(prompt).content
        if not isinstance(revised, str):
            revised = str(revised)
    except Exception:
        revised = state.get("draft_answer", "") + "\n\nRevision note: " + state.get("critique", "Review feedback unavailable.")
    return {"draft_answer": revised, "revision_attempts": state.get("revision_attempts", 0) + 1, "audit_log": ["Revision agent updated draft"]}


def build_multi_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", _route_node)
    graph.add_node("profiler", _specialist_node("profiler"))
    graph.add_node("metrics", _specialist_node("metrics"))
    graph.add_node("trends", _specialist_node("trends"))
    graph.add_node("anomalies", _specialist_node("anomalies"))
    graph.add_node("synthesize", _synthesize_node)
    graph.add_node("critic", _critic_node)
    graph.add_node("revise", _revise_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", _next_specialist, {**{name: name for name in SPECIALISTS}, "synthesize": "synthesize"})
    for name in SPECIALISTS:
        graph.add_edge(name, "router_dispatch")
    # A lightweight dispatch node uses the updated cursor to choose the next specialist.
    graph.add_node("router_dispatch", lambda state: {"audit_log": []})
    graph.add_conditional_edges("router_dispatch", _next_specialist, {**{name: name for name in SPECIALISTS}, "synthesize": "synthesize"})
    graph.add_edge("synthesize", "critic")
    graph.add_conditional_edges("critic", _critic_route, {"revise": "revise", END: END})
    graph.add_edge("revise", "critic")
    return graph.compile()


def run_multi_agent(df: pd.DataFrame, question: str) -> dict:
    """Run the multi-agent graph with a compact schema/profile snapshot."""
    numeric = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [str(c) for c in df.columns if c not in numeric and not pd.api.types.is_datetime64_any_dtype(df[c])]
    date_candidates = [str(c) for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or any(token in str(c).lower() for token in ("date", "time", "month"))]
    summary = {
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "date_candidates": date_candidates,
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items()},
    }
    initial: AgentState = {
        "question": question,
        "columns": list(map(str, df.columns)),
        "data_summary": summary,
        "specialist_cursor": 0,
        "findings": [],
        "audit_log": [],
        "revision_attempts": 0,
    }
    return build_multi_agent_graph().invoke(initial)
