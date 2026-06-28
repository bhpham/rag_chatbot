
"""
YOUR CODE HERE
"""

import pathlib, streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Customer Support Chatbot")
st.title("Customer Support Chatbot")

SYSTEM_TEMPLATE = """
You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with "I'm not sure from the docs."

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
4) When possible, cite sources as [source: source] using the metadata.

CONTEXT:
{context}

USER:
{question}
"""

@st.cache_resource
def init_resources():
    vectordb = FAISS.load_local(
        "faiss_index",
        HuggingFaceEmbeddings(model_name="thenlper/gte-small"),
        allow_dangerous_deserialization=True,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    llm = OllamaLLM(model="gemma3:1b", temperature=0.1)
    prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
    return retriever, llm, prompt

retriever, llm, prompt = init_resources()

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def rag_step(question, chat_history):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    messages = prompt.format_messages(context=context, question=question)
    answer = llm.invoke(messages)
    return answer

if "history" not in st.session_state:
    st.session_state.history = []

for user, bot in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(bot)

question = st.chat_input("What is on your mind?")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_step(question, st.session_state.history)
        st.markdown(answer)
    st.session_state.history.append((question, answer))
