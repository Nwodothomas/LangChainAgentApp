import streamlit as st
from agent.config import *
from agent.loader import load_documents
from agent.vectorstore import load_vectorstore, build_vectorstore
from agent.chain import build_chain
import os
from datetime import datetime

# Page setup
st.set_page_config(
    page_title="Cardiovascular Study Assistant", 
    layout="wide",
    page_icon="❤️"
)

# Paths
docs_path = "data/docs"
persist_path = "vectorstore"

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_query" not in st.session_state:
    st.session_state.processed_query = ""

# Custom CSS for professional styling
st.markdown("""
    <style>
        .main-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        .header-section {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            color: white;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-bottom: 30px;
        }
        .user-message {
            align-self: flex-end;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 18px 18px 5px 18px;
            max-width: 70%;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .assistant-message {
            align-self: flex-start;
            background: #f8f9fa;
            color: #333;
            padding: 15px 20px;
            border-radius: 18px 18px 18px 5px;
            max-width: 70%;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .message-header {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
            font-weight: 600;
        }
        .avatar {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .timestamp {
            font-size: 0.7rem;
            opacity: 0.7;
            margin-top: 8px;
            text-align: right;
        }
        .input-section {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 90%;
            max-width: 800px;
            background: white;
            padding: 15px;
            border-radius: 25px;
            box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        .stButton button {
            width: 100%;
            border-radius: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            font-weight: 600;
        }
        .upload-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .warning {
            background: #fff3cd;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for file upload
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2>📄 Document Management</h2>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📤 Upload Documents", expanded=True):
        st.markdown("""
            <div class="upload-section">
                <p><strong>Supported formats:</strong> PDF, DOCX, TXT</p>
                <p><strong>Size limit:</strong> 200MB per file</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            os.makedirs(docs_path, exist_ok=True)
            for file in uploaded_files:
                file_path = os.path.join(docs_path, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            st.info("🔄 Please refresh the page to re-index documents.")

# Main content area
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header section
st.markdown("""
    <div class="header-section">
        <h1>❤️ Cardiovascular Study Assistant</h1>
        <p>Ask questions about cardiovascular research, biomarkers, and clinical studies</p>
    </div>
""", unsafe_allow_html=True)

# Load or build vectorstore
try:
    if not os.path.exists(f"{persist_path}/index.faiss"):
        with st.spinner("🔍 Indexing documents... This may take a few moments."):
            docs = load_documents(docs_path)
            vectorstore = build_vectorstore(docs, persist_path)
            st.success("✅ Documents indexed successfully!")
    else:
        vectorstore = load_vectorstore(persist_path)
    
    # Build QA chain
    qa_chain = build_chain(vectorstore)
    
except Exception as e:
    st.error(f"❌ Error initializing the assistant: {str(e)}")
    st.info("💡 Please make sure you have uploaded some documents first.")
    qa_chain = None

# Chat history display
st.markdown("### 💬 Conversation History")
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown("""
        <div style="text-align: center; padding: 40px; color: #6c757d;">
            <h3>👋 Welcome to the Cardiovascular Study Assistant!</h3>
            <p>Start a conversation by asking a question below.</p>
            <p><strong>Example questions:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li>What biomarkers were most predictive of cardiovascular disease?</li>
                <li>Explain the relationship between cholesterol and heart disease</li>
                <li>What are the latest treatments for hypertension?</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
else:
    for i, chat in enumerate(st.session_state.chat_history):
        # User message
        st.markdown(f"""
            <div class="user-message">
                <div class="message-header">
                    <img src="https://api.dicebear.com/6.x/personas/svg?seed=user{i}" class="avatar">
                    You
                </div>
                {chat['question']}
                <div class="timestamp">{chat['timestamp']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Assistant message
        st.markdown(f"""
            <div class="assistant-message">
                <div class="message-header">
                    <img src="https://api.dicebear.com/6.x/bottts/svg?seed=assistant{i}" class="avatar">
                    Assistant
                </div>
                {chat['answer']}
                <div class="timestamp">{chat['timestamp']}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "Ask a question about cardiovascular studies...",
        key="query_input",
        label_visibility="collapsed",
        placeholder="e.g., What biomarkers were most predictive of cardiovascular disease?"
    )

with col2:
    submit_btn = st.button("Send", use_container_width=True)

# Voice input note
st.markdown("""
    <div style="text-align: center; margin-top: 60px; color: #6c757d;">
        <small>🎙️ You can use voice input by clicking the mic icon in your browser (if supported)</small>
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close input-section
st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

# Handle query processing
if submit_btn and query and query != st.session_state.processed_query:
    if qa_chain is None:
        st.error("❌ Assistant is not ready. Please check if documents are properly loaded.")
    else:
        with st.spinner("🤔 Analyzing your question..."):
            try:
                response = qa_chain.invoke(query)
                answer = response.get("result", "I couldn't generate a response based on the available documents.")
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer,
                    "timestamp": timestamp
                })
                st.session_state.processed_query = query
                
                # Force rerun to display the new message
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing your question: {str(e)}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": f"Sorry, I encountered an error while processing your request: {str(e)}",
                    "timestamp": timestamp
                })
                st.session_state.processed_query = query
                st.rerun()

# Add some padding at the bottom for better mobile experience
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)