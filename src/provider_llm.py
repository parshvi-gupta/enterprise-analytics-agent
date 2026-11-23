"""Bring-your-own-key integrations. Keys are not persisted by this app."""
from __future__ import annotations
import json

def ask_provider(provider: str, api_key: str, question: str, findings: list[dict]) -> str:
    evidence = json.dumps(findings, default=str)[:12000]
    prompt = (
        "You are an enterprise analytics assistant. Interpret ONLY the provided deterministic evidence. "
        "Do not invent figures or claim to have directly inspected the dataset. Mention limitations if evidence is insufficient. "
        "Be concise and answer the user's business question.\n\n"
        f"Question: {question}\n\nDeterministic evidence from analytics tools:\n{evidence}"
    )
    if provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or "No response returned."
    if provider == "Google Gemini":
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or "No response returned."
    if provider == "Anthropic Claude":
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=900,
            messages=[{"role":"user","content":prompt}],
        )
        return "".join(block.text for block in response.content if getattr(block,"type",None)=="text") or "No response returned."
    raise ValueError("Unsupported provider selected.")
