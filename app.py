import streamlit as st
from agent.config import *
from agent.loader import load_documents
from agent.vectorstore import load_vectorstore, build_vectorstore
from agent.chain import build_chain
import os
from datetime import datetime

st.set_page_config(page_title="Cardiovascular Study Assistant", layout="wide")
st.title("🧠 Cardiovascular Study Assistant")

persist_path = "vectorstore"
docs_path = "data/docs"

# Upload new documents
uploaded_files = st.file_uploader("Upload new study documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(docs_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success("Files uploaded! Please refresh to re-index.")

# Load or build vectorstore
if not os.path.exists(f"{persist_path}/index.faiss"):
    with st.spinner("Indexing documents..."):
        docs = load_documents(docs_path)
        vectorstore = build_vectorstore(docs, persist_path)
else:
    vectorstore = load_vectorstore(persist_path)

qa_chain = build_chain(vectorstore)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Voice input (browser mic)
st.markdown("🎙️ You can use voice input by clicking the mic icon in your browser (if supported).")

# Input field
query = st.text_input("Ask a question about the study:", key="user_input")

# Handle query
if query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke(query)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({
            "question": query,
            "answer": response["result"],
            "timestamp": timestamp
        })
        st.session_state.user_input = ""  # Clear input

# Display chat history
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"""
    <div style="background-color:#f1f1f1; padding:10px; border-radius:10px; margin-bottom:10px;">
        <div style="display:flex; align-items:center;">
            <img src="https://avatars.githubusercontent.com/u/1?v=4" width="30" style="border-radius:50%; margin-right:10px;">
            <strong>You</strong> <span style="color:gray; font-size:12px;">{chat['timestamp']}</span>
        </div>
        <div style="margin-top:5px;">{chat['question']}</div>
    </div>
    <div style="background-color:#e0f7fa; padding:10px; border-radius:10px; margin-bottom:20px;">
        <div style="display:flex; align-items:center;">
            <img src="https://avatars.githubusercontent.com/u/2?v=4" width="30" style="border-radius:50%; margin-right:10px;">
            <strong>Agent</strong> <span style="color:gray; font-size:12px;">{chat['timestamp']}</span>
        </div>
        <div style="margin-top:5px;">{chat['answer']}</div>
    </div>
    """, unsafe_allow_html=True)