"""Generation part: persona prompt + Groq chat API (OpenAI-compatible)."""
import os

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

PERSONA_PROMPT = """You are the AI clone of Fatima, a software engineering student from Sialkot, Pakistan.
Speak in the first person ("I"), in a casual, friendly tone. Keep answers short (2-4 sentences).

Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. If the context does not contain the answer, say: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, addresses, ID numbers or other private details.
4. If asked whether you are the real person, say you are an AI clone of Fatima.
5. Reply in the same language the question is asked in (English or Roman Urdu).

CONTEXT:
{context}
"""


def build_context(hits):
    return "\n\n".join(
        f"[{h['chunk']['source']} / {h['chunk']['chunk_id']}]\n{h['chunk']['text']}"
        for h in hits
    )


def fallback_answer(hits):
    """Templated answer used when no LLM is available."""
    if not hits:
        return "I don't have that in my notes."
    snippet = hits[0]["chunk"]["text"][:400]
    return f"(LLM unavailable) Most relevant note: {snippet}..."


def generate_answer(question, hits, api_key):
    """Returns (answer, error_message_or_None)."""
    if not hits:
        return "I don't have that in my notes.", None
    if not api_key:
        return fallback_answer(hits), "GROQ_API_KEY not found. Add it as a secret."
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.2,
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": question},
        ],
    }
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=30,
        )
        if r.status_code != 200:
            return fallback_answer(hits), f"Groq error {r.status_code}: {r.text[:300]}"
        return r.json()["choices"][0]["message"]["content"].strip(), None
    except Exception as e:  # network errors, bad JSON, ...
        return fallback_answer(hits), f"LLM call failed: {e}"
