"""
app.py - Streamlit Web Interface for Document Q&A Chatbot
"""

import sys
sys.path.insert(0, '.')

import os
import tempfile
import streamlit as st
from src.rag import RAGSystem
from src.graph import QAWorkflow
from src.memory import MemorySystem

st.set_page_config(page_title="Document Q&A Chatbot", page_icon="💬", layout="centered")
st.title("Document Q&A Chatbot")
st.caption("Powered by RAG + Memory + LangGraph + Groq")

# --- Initialize RAG and workflow once ---
@st.cache_resource(show_spinner=False)
def load_systems():
    rag = RAGSystem()
    try:
        count = rag.load_documents()
        status = f"Loaded {count} chunks from docs/ folder."
    except FileNotFoundError:
        status = ""
    except Exception as e:
        status = f"Warning: {e}"
    workflow = QAWorkflow()
    return rag, workflow, status

with st.spinner("Starting up... (first run downloads embedding model ~90MB)"):
    try:
        rag, workflow, load_status = load_systems()
        if load_status:
            st.info(load_status)
    except Exception as e:
        st.error(f"Startup error: {e}")
        st.stop()

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-chat"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Documents")

    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        for file in uploaded:
            if file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        # Save to temp file then load into RAG
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name

                        chunks = rag.load_pdf(tmp_path)
                        os.unlink(tmp_path)

                        st.session_state.uploaded_files.append(file.name)
                        st.success(f"{file.name}: {chunks} chunks added")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")

    if st.session_state.uploaded_files:
        st.markdown("**Loaded PDFs:**")
        for name in st.session_state.uploaded_files:
            st.markdown(f"- {name}")

    st.markdown("---")
    if st.button("Clear Memory"):
        MemorySystem().clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.success("Memory cleared!")

    st.markdown("---")
    st.markdown("**Model:** llama-3.1-8b-instant (Groq)")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2")

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
