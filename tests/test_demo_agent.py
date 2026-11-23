import pandas as pd
from src.demo_agent import route_question, run_demo_analysis

def test_routing_detects_metrics_and_trends():
    assert "metrics" in route_question("compare revenue by region")
    assert "trends" in route_question("monthly sales trend")

def test_demo_analysis_needs_no_api_key():
    df = pd.DataFrame({"region":["East","West"],"revenue":[100,150]})
    result = run_demo_analysis(df, "compare revenue by region")
    assert result["final_answer"].startswith("### Demo-mode analysis")
    assert result["specialists"]
    assert result["findings"][0]["evidence"]


def test_profiler_reports_schema_and_missing_cells():
    df = pd.DataFrame({"name":["A",None],"sales":[10,20]})
    result = run_demo_analysis(df, "profile dataset")
    profiler = next(x for x in result["findings"] if x["specialist"] == "profiler")
    assert profiler["evidence"]["rows"] == 2
    assert profiler["evidence"]["columns"] == ["name", "sales"]
    assert sum(profiler["evidence"]["missing_values"].values()) == 1
