"""
app.py - Streamlit Web Interface for Document Q&A Chatbot
"""

import sys
sys.path.insert(0, '.')

import streamlit as st
from src.rag import RAGSystem
from src.graph import QAWorkflow
from src.memory import MemorySystem

st.set_page_config(page_title="Document Q&A Chatbot", page_icon="💬", layout="centered")
st.title("Document Q&A Chatbot")
st.caption("Powered by RAG + Memory + LangGraph + Ollama (tinyllama)")

# --- Initialize systems (only once per session) ---
@st.cache_resource(show_spinner=False)
def load_systems():
    rag = RAGSystem()
    try:
        count = rag.load_documents()
        status = f"Loaded {count} document chunks."
    except FileNotFoundError:
        status = "Warning: No documents folder found. Add .txt files to the docs/ folder."
    except Exception as e:
        status = f"Warning loading documents: {e}"
    workflow = QAWorkflow()
    return workflow, status

with st.spinner("Loading documents and models... (first run may take a minute)"):
    try:
        workflow, load_status = load_systems()
        if "Warning" in load_status:
            st.warning(load_status)
        else:
            st.success(load_status)
    except Exception as e:
        st.error(f"Startup error: {e}")
        st.stop()

# --- Session state for chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-chat"

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Memory"):
        MemorySystem().clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.success("Memory cleared!")
    st.markdown("---")
    st.markdown("**Model:** tinyllama (via Ollama)")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2")
    st.markdown("**Documents:** docs/*.txt")

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = workflow.run(question, st.session_state.session_id)
            except Exception as e:
                answer = f"Error: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
