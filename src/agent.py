"""Bounded LLM tool-calling agent. No arbitrary code execution is exposed."""
from __future__ import annotations
import json
import os
from typing import Any
import pandas as pd
from openai import OpenAI
from .analytics_tools import execute_tool, tool_specs

SYSTEM_PROMPT = """You are a business analytics assistant. Use the available tools to compute answers from the uploaded dataset. Never invent data. State the metric and filters used. If the data cannot answer the question, say so. Explain insights concisely and distinguish observations from hypotheses."""


def ask_agent(df: pd.DataFrame, question: str, max_steps: int = 5) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Configure it in your environment or .env file.")
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT + f" Dataset columns: {list(df.columns)}"},
        {"role": "user", "content": question},
    ]
    tools = tool_specs(df)
    for _ in range(max_steps):
        response = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto", temperature=0)
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content or "The model returned an empty response."
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(df, call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})
    return "Stopped after the maximum number of tool iterations. Try a narrower question."
