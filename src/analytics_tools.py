"""Safe, deterministic analytics tools available to the agent."""
from __future__ import annotations
import json
from typing import Any
import pandas as pd


def profile_data(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
        "numeric_summary": df.describe(include="number").round(2).to_dict(),
    }


def grouped_metric(df: pd.DataFrame, group_by: str, metric: str, operation: str = "sum") -> dict[str, Any]:
    if group_by not in df.columns or metric not in df.columns:
        return {"error": "Requested column does not exist", "available_columns": list(df.columns)}
    if operation not in {"sum", "mean", "median", "count", "min", "max"}:
        return {"error": "Unsupported operation"}
    if operation != "count" and not pd.api.types.is_numeric_dtype(df[metric]):
        return {"error": f"Metric '{metric}' must be numeric for {operation}"}
    result = df.groupby(group_by, dropna=False)[metric].agg(operation).sort_values(ascending=False)
    return {"group_by": group_by, "metric": metric, "operation": operation, "results": result.to_dict()}


def monthly_trend(df: pd.DataFrame, date_col: str, metric: str, operation: str = "sum") -> dict[str, Any]:
    if date_col not in df.columns or metric not in df.columns:
        return {"error": "Date or metric column not found"}
    if operation not in {"sum", "mean", "count"}:
        return {"error": "Supported operations: sum, mean, count"}
    dates = pd.to_datetime(df[date_col], errors="coerce")
    valid = df.loc[dates.notna()].copy()
    valid["_month"] = dates.loc[dates.notna()].dt.to_period("M").astype(str)
    if operation != "count" and not pd.api.types.is_numeric_dtype(valid[metric]):
        return {"error": "Metric must be numeric"}
    return {"trend": valid.groupby("_month")[metric].agg(operation).to_dict()}


def anomaly_scan(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    if metric not in df.columns or not pd.api.types.is_numeric_dtype(df[metric]):
        return {"error": "Choose an existing numeric column"}
    values = df[metric].dropna()
    if values.empty:
        return {"anomalies": [], "method": "IQR", "note": "No numeric values found"}
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = df.loc[(df[metric] < lower) | (df[metric] > upper)].copy()
    return {"method": "IQR", "bounds": {"lower": float(lower), "upper": float(upper)}, "anomalies": flagged.head(50).to_dict(orient="records")}



def tool_specs(df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = list(df.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return [
        {"type": "function", "function": {"name": "profile_data", "description": "Profile dataset schema, missingness, and numeric summaries.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
        {"type": "function", "function": {"name": "grouped_metric", "description": "Aggregate a numeric column by a categorical column.", "parameters": {"type": "object", "properties": {"group_by": {"type": "string", "enum": cols}, "metric": {"type": "string", "enum": cols}, "operation": {"type": "string", "enum": ["sum","mean","median","count","min","max"]}}, "required": ["group_by","metric","operation"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "monthly_trend", "description": "Aggregate a metric by month using a date column.", "parameters": {"type": "object", "properties": {"date_col": {"type": "string", "enum": cols}, "metric": {"type": "string", "enum": cols}, "operation": {"type": "string", "enum": ["sum","mean","count"]}}, "required": ["date_col","metric","operation"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "anomaly_scan", "description": "Find outliers in a numeric column using IQR fences.", "parameters": {"type": "object", "properties": {"metric": {"type": "string", "enum": numeric}}, "required": ["metric"], "additionalProperties": False}}},
    ]


def execute_tool(df: pd.DataFrame, name: str, args: dict[str, Any]) -> dict[str, Any]:
    allowed = {"profile_data", "grouped_metric", "monthly_trend", "anomaly_scan"}
    if name not in allowed:
        return {"error": "Tool not permitted"}
    try:
        if name == "profile_data": return profile_data(df)
        if name == "grouped_metric": return grouped_metric(df, **args)
        if name == "monthly_trend": return monthly_trend(df, **args)
        if name == "anomaly_scan": return anomaly_scan(df, **args)
    except (TypeError, ValueError, KeyError) as exc:
        return {"error": str(exc)}
    return {"error": "Unknown tool"}
