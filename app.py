import streamlit as st
from agent.config import *
from agent.loader import load_documents
from agent.vectorstore import load_vectorstore, build_vectorstore
from agent.chain import build_chain
import os
from datetime import datetime

# Page setup
st.set_page_config(page_title="Cardiovascular Study Assistant", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for ChatGPT-style UI
st.markdown("""
    <style>
        .chat-container {
            max-width: 800px;
            margin: auto;
        }
        .chat-bubble {
            padding: 1rem;
            border-radius: 1rem;
            margin-bottom: 1rem;
            max-width: 90%;
        }
        .user-bubble {
            background-color: #f0f0f0;
            align-self: flex-end;
        }
        .agent-bubble {
            background-color: #e0f7fa;
            align-self: flex-start;
        }
        .avatar {
            width: 30px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .chat-row {
            display: flex;
            align-items: flex-start;
        }
        .timestamp {
            font-size: 0.75rem;
            color: gray;
            margin-top: 4px;
        }
        .input-bar {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 20px;
        }
        .input-box {
            flex-grow: 1;
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .send-button {
            background-color: #00aaff;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
        }
        .send-button:hover {
            background-color: #008ecc;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.markdown("## 🧠 Cardiovascular Study Assistant")

# File upload section
docs_path = "data/docs"
persist_path = "vectorstore"
uploaded_files = st.file_uploader("📄 Upload new study documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
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

# Handle query
def handle_query():
    query = st.session_state.user_input
    if query:
        with st.spinner("🤔 Thinking..."):
            response = qa_chain.invoke(query)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append({
                "question": query,
                "answer": response["result"],
                "timestamp": timestamp
            })
            st.session_state.user_input = ""  # Clear input

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(f"""
        <div class="chat-row">
            <img src="https://avatars.githubusercontent.com/u/1?v=4" class="avatar">
            <div class="chat-bubble user-bubble">
                <strong>You</strong><br>{chat['question']}
                <div class="timestamp">{chat['timestamp']}</div>
            </div>
        </div>
        <div class="chat-row">
            <img src="https://avatars.githubusercontent.com/u/2?v=4" class="avatar">
            <div class="chat-bubble agent-bubble">
                <strong>Agent</strong><br>{chat['answer']}
                <div class="timestamp">{chat['timestamp']}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Voice input note
st.markdown("🎙️ You can use voice input by clicking the mic icon in your browser (if supported).")

# Input bar
st.markdown("<div class='input-bar'>", unsafe_allow_html=True)
st.text_input("Ask a question...", key="user_input", label_visibility="collapsed", on_change=handle_query)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
