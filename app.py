"""Fatima's Digital Clone - Streamlit app (Hybrid RAG: BM25 + dense + RRF + Groq)."""
import os

import streamlit as st

from llm import GROQ_MODEL, generate_answer
from rag import MODE_HYBRID, MODES, Retriever

st.set_page_config(page_title="Fatima's Digital Clone", page_icon="🎨", layout="wide")


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_retriever():
    return Retriever()


def get_api_key():
    key = os.environ.get("GROQ_API_KEY")  # Hugging Face Space secret
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]  # Streamlit secrets
    except Exception:
        return None


st.title("🎨 Fatima's Digital Clone")
st.warning("🤖 This is an AI clone, not the real person.")
st.caption("Ask me about my studies, hobbies, projects... I answer only from my own notes.")

with st.sidebar:
    st.header("Retrieval settings")
    mode = st.radio("Retrieval mode", MODES, index=MODES.index(MODE_HYBRID))
    top_k = st.slider("Chunks sent to the LLM (top-k)", 1, 8, 4)
    st.caption(f"LLM: {GROQ_MODEL} via Groq")

retriever = get_retriever()

with st.form("ask"):
    question = st.text_input("Your question", placeholder="e.g. What are your hobbies?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    hits = retriever.search(question, mode, top_k)
    with st.spinner("Thinking..."):
        answer, error = generate_answer(question, hits, get_api_key())

    st.subheader("Answer")
    st.write(answer)
    if error:
        with st.expander("LLM error details"):
            st.code(error)

    st.subheader(f"Retrieved chunks ({mode})")
    if not hits:
        st.info("Nothing relevant was retrieved for this question.")
    for rank, h in enumerate(hits, start=1):
        c = h["chunk"]
        title = f"#{rank}  {c['source']}  |  {c['chunk_id']}  |  score {h['score']:.4f}"
        with st.expander(title, expanded=(rank == 1)):
            if mode == MODE_HYBRID:
                st.caption(f"BM25 rank: {h['bm25_rank'] or '-'}   |   Dense rank: {h['dense_rank'] or '-'}")
            st.write(c["text"])
