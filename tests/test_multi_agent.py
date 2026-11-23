import pandas as pd
import pytest
from src.multi_agent import fallback_route, _tool_evidence, build_multi_agent_graph

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
        "region": ["East", "East", "West"],
        "revenue": [100, 150, 300],
        "units": [10, 15, 30],
    })

def test_fallback_routes_metrics():
    assert "metrics" in fallback_route("Which region has highest revenue?")

def test_fallback_routes_trends():
    assert "trends" in fallback_route("Show monthly trend over time")

def test_fallback_routes_anomalies():
    assert "anomalies" in fallback_route("Find unusual revenue outliers")

def test_metrics_specialist_computes_real_grouped_values(sample_df):
    evidence = _tool_evidence(sample_df, "metrics", "Compare revenue by region")
    assert evidence["results"]["East"] == 250
    assert evidence["results"]["West"] == 300

def test_trend_specialist_computes_real_months(sample_df):
    evidence = _tool_evidence(sample_df, "trends", "monthly revenue trend")
    assert evidence["trend"]["2025-02"] == 150

def test_anomaly_specialist_returns_iqr_evidence(sample_df):
    evidence = _tool_evidence(sample_df, "anomalies", "detect revenue anomalies")
    assert evidence["method"] == "IQR"
    assert "bounds" in evidence

def test_graph_compiles_without_api_key(sample_df):
    compiled = build_multi_agent_graph(sample_df)
    assert compiled is not None
