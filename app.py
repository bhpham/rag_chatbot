"""Everstorm customer support chatbot.

Retrieval settings live in eval/config.py, the same file the eval harness
scores, so a configuration that wins on the benchmark ships here with no
manual copying.
"""

import sys
from pathlib import Path

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rag_core
from eval.config import CONFIG

st.set_page_config(page_title="Customer Support Chatbot")
st.title("Customer Support Chatbot")


@st.cache_resource
def init_resources():
    retriever = rag_core.get_retriever(CONFIG)
    llm = rag_core.get_llm(CONFIG)
    prompt = ChatPromptTemplate.from_template(CONFIG["system_template"])
    return retriever, llm, prompt


retriever, llm, prompt = init_resources()

if "history" not in st.session_state:
    st.session_state.history = []


def render_sources(docs):
    """Show what the answer was actually built from.

    Deterministic attribution, independent of whether the model followed the
    citation instruction in the prompt.
    """
    with st.expander(f"Sources ({len(docs)} chunks retrieved)"):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**{i}. {rag_core.source_label(doc)}**")
            st.caption(doc.page_content.strip()[:400])


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        render_sources(turn["docs"])

question = st.chat_input("What is on your mind?")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag_core.answer(
                question, CONFIG, retriever=retriever, llm=llm, prompt=prompt
            )
        st.markdown(result["answer"])
        render_sources(result["docs"])
    st.session_state.history.append(
        {"question": question, "answer": result["answer"], "docs": result["docs"]}
    )
