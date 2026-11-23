# Enterprise Analytics Agent

An agentic analytics copilot that converts natural-language business questions into a plan, executes data analysis tools, validates results, and returns evidence-backed insights.

## MVP capabilities
- Upload CSV data or use the included sample sales dataset
- Ask business questions in natural language
- LLM tool-calling loop with bounded execution
- Tools: dataset profile, grouped metrics, trend analysis, anomaly scan
- Lightweight Streamlit UI
- Clear separation of data tools, agent orchestration, and UI

> This project is a portfolio demo. Use synthetic/sample data only. The agent is not connected to production systems.

## Architecture
```
Question -> Agent loop -> Tool selection -> Analytics tools -> Tool result
                  ^                              |
                  +---------- iteration ----------+
                              |
                       Answer + caveats
```

## Stack
Python 3.11+, Streamlit, pandas, OpenAI API (optional to run the agent), pytest.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY if using the LLM mode
streamlit run app.py
```

## Demo questions
- Which region has the highest revenue?
- Compare revenue by channel.
- Show monthly sales trend.
- Find unusual revenue values.

Without an API key, the app still supports dataset profiling and the built-in deterministic analytics functions.

## Safety and reliability
- Tool allow-list (no arbitrary code execution)
- Tool call iteration cap
- Tool input validation
- Errors returned to the model as tool results
- Synthetic sample dataset

## Roadmap
- LangGraph workflow with planner / analyst / critic nodes
- DuckDB SQL analytics
- RAG over metric definitions
- Evaluation dataset and tracing
- Docker and CI

## License
MIT
