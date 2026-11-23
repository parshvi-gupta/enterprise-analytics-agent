import pandas as pd
import pytest
from src.analytics_tools import profile_data, grouped_metric, monthly_trend, anomaly_scan, execute_tool


@pytest.fixture
def sample_df():
    return pd.DataFrame({"date": ["2025-01-01", "2025-02-01", "2025-03-01"], "region": ["A", "A", "B"], "revenue": [10, 20, 100]})


def test_profile_counts(sample_df):
    p = profile_data(sample_df)
    assert p["rows"] == 3
    assert "revenue" in p["columns"]


def test_grouped_metric_sum(sample_df):
    result = grouped_metric(sample_df, "region", "revenue", "sum")
    assert result["results"]["A"] == 30
    assert result["results"]["B"] == 100


def test_monthly_trend(sample_df):
    result = monthly_trend(sample_df, "date", "revenue")
    assert result["trend"]["2025-02"] == 20


def test_anomaly_scan_returns_shape(sample_df):
    result = anomaly_scan(sample_df, "revenue")
    assert "anomalies" in result and "method" in result


def test_tool_allowlist(sample_df):
    assert execute_tool(sample_df, "os.system", {})["error"] == "Tool not permitted"
