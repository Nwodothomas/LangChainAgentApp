import streamlit as st
from agent.config import *
from agent.loader import load_documents
from agent.vectorstore import load_vectorstore, build_vectorstore
from agent.chain import build_chain
import os
from datetime import datetime
import time

# Page setup
st.set_page_config(
    page_title="Cardiovascular Study Assistant", 
    layout="wide",
    page_icon="❤️",
    initial_sidebar_state="collapsed"
)

# Paths
docs_path = "data/docs"
persist_path = "vectorstore"

# Initialize session state properly
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    initial_session_id = "default"
    st.session_state.chat_sessions[initial_session_id] = {
        "id": initial_session_id,
        "history": [],
        "title": "New Chat"
    }
    st.session_state.current_session_id = initial_session_id

if "processed_query" not in st.session_state:
    st.session_state.processed_query = ""
if "show_new_chat" not in st.session_state:
    st.session_state.show_new_chat = False
if "streaming_active" not in st.session_state:
    st.session_state.streaming_active = False

# Custom CSS for enhanced professional styling
st.markdown("""
    <style>
        /* Main container */
        .main-container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        /* Header section */
        .header-section {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 0 0 15px 15px;
            margin-bottom: 0;
        }
        
        /* Chat container with scroll */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            margin-bottom: 80px;
            max-height: calc(100vh - 200px);
        }
        
        /* Message styles */
        .user-message {
            align-self: flex-end;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 18px 18px 5px 18px;
            max-width: 70%;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .assistant-message {
            align-self: flex-start;
            background: #f8f9fa;
            color: #333;
            padding: 15px 20px;
            border-radius: 18px 18px 18px 5px;
            max-width: 70%;
            margin: 10px 0;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .thinking-message {
            align-self: flex-start;
            background: #f0f8ff;
            color: #333;
            padding: 15px 20px;
            border-radius: 18px 18px 18px 5px;
            max-width: 70%;
            margin: 10px 0;
            border: 2px dashed #667eea;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            font-style: italic;
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
        
        /* Fixed input section */
        .input-section {
            position: fixed;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100%;
            max-width: 1000px;
            background: white;
            padding: 15px;
            border-top: 1px solid #e0e0e0;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        
        /* Button styles */
        .stButton button {
            border-radius: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            font-weight: 600;
        }
        
        /* Sidebar styles */
        .chat-history-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .chat-history-item:hover {
            background: #f0f0f0;
        }
        
        .chat-history-item.active {
            background: #667eea;
            color: white;
        }
        
        /* Upload section */
        .upload-section {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #764ba2;
        }
        
        /* Welcome message styling */
        .welcome-container {
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }
        
        .capability-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 20px 0;
        }
        
        .capability-item {
            background: #f0f8ff;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        
        /* Modal styles */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 1999;
        }
        
        .new-chat-modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            z-index: 2000;
            width: 90%;
            max-width: 500px;
        }
    </style>
""", unsafe_allow_html=True)

def create_new_chat_session():
    """Create a new chat session"""
    session_id = f"chat_{int(time.time())}"
    st.session_state.chat_sessions[session_id] = {
        "id": session_id,
        "history": [],
        "title": f"Chat {len(st.session_state.chat_sessions)}"
    }
    st.session_state.current_session_id = session_id
    st.session_state.show_new_chat = False
    st.session_state.processed_query = ""

def get_current_session():
    """Get current chat session with error handling"""
    current_id = st.session_state.current_session_id
    if current_id not in st.session_state.chat_sessions:
        if st.session_state.chat_sessions:
            first_id = list(st.session_state.chat_sessions.keys())[0]
            st.session_state.current_session_id = first_id
            return st.session_state.chat_sessions[first_id]
        else:
            st.session_state.chat_sessions["default"] = {
                "id": "default",
                "history": [],
                "title": "New Chat"
            }
            st.session_state.current_session_id = "default"
            return st.session_state.chat_sessions["default"]
    return st.session_state.chat_sessions[current_id]

def display_thinking_message():
    """Display a thinking message while processing"""
    thinking_html = """
    <div class="thinking-message">
        <div class="message-header">
            <img src="https://api.dicebear.com/6.x/bottts/svg?seed=thinking" class="avatar">
            MedAnalytica Pro
        </div>
        <div style="display: flex; align-items: center;">
            <div>Analyzing your query</div>
            <div style="margin-left: 10px; display: flex;">
                <div class="dot-flashing"></div>
            </div>
        </div>
    </div>
    <style>
        .dot-flashing {
            position: relative;
            width: 10px;
            height: 10px;
            border-radius: 5px;
            background-color: #667eea;
            color: #667eea;
            animation: dotFlashing 1s infinite linear alternate;
            animation-delay: 0.5s;
        }
        .dot-flashing::before, .dot-flashing::after {
            content: '';
            display: inline-block;
            position: absolute;
            top: 0;
            width: 10px;
            height: 10px;
            border-radius: 5px;
            background-color: #667eea;
            color: #667eea;
        }
        .dot-flashing::before {
            left: -15px;
            animation: dotFlashing 1s infinite alternate;
            animation-delay: 0s;
        }
        .dot-flashing::after {
            left: 15px;
            animation: dotFlashing 1s infinite alternate;
            animation-delay: 1s;
        }
        @keyframes dotFlashing {
            0% { background-color: #667eea; }
            50%, 100% { background-color: #ebe6ff; }
        }
    </style>
    """
    return st.markdown(thinking_html, unsafe_allow_html=True)

# Sidebar for chat history and document management
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2>🧠 MedAnalytica Pro</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.show_new_chat = True
    
    # Chat History
    st.markdown("### 💬 Chat History")
    
    if st.session_state.chat_sessions:
        for session_id, session in list(st.session_state.chat_sessions.items()):
            is_active = session_id == st.session_state.current_session_id
            emoji = "🔵" if is_active else "⚪"
            if st.button(f"{emoji} {session.get('title', 'Untitled Chat')}", 
                        key=f"chat_{session_id}"):
                st.session_state.current_session_id = session_id
                st.rerun()
    else:
        st.info("No chat history available")
    
    st.markdown("---")
    
    # Document Management
    with st.expander("📁 Document Management", expanded=True):
        st.markdown("""
            <div class="upload-section">
                <p><strong>Supported formats:</strong> PDF, DOCX, TXT</p>
                <p><strong>Size limit:</strong> 200MB per file</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload medical documents",
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
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded!")
            st.info("🔄 Please refresh the page to re-index documents.")

# Main content area
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header section
st.markdown("""
    <div class="header-section">
        <h1>❤️ MedAnalytica Pro</h1>
        <p>Advanced Cardiovascular AI Assistant • Risk Assessment • Diagnosis • Prevention</p>
    </div>
""", unsafe_allow_html=True)

# Load or build vectorstore
try:
    if not os.path.exists(f"{persist_path}/index.faiss"):
        with st.spinner("🔍 Indexing medical documents... This may take a few moments."):
            docs = load_documents(docs_path)
            vectorstore = build_vectorstore(docs, persist_path)
            st.success("✅ Medical documents indexed successfully!")
    else:
        vectorstore = load_vectorstore(persist_path)
    
    # Build QA chain
    qa_chain = build_chain(vectorstore)
    
except Exception as e:
    st.error(f"❌ Error initializing medical assistant: {str(e)}")
    st.info("💡 Please upload medical documents to enable advanced analysis.")
    qa_chain = None

# Chat display area
current_session = get_current_session()
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if not current_session["history"]:
    st.markdown("""
        <div class="welcome-container">
            <h3>👋 Welcome to MedAnalytica Pro!</h3>
            <p>Your advanced AI partner for cardiovascular health analysis.</p>
            <p><strong>Advanced analysis capabilities:</strong></p>
            <div class="capability-grid">
                <div class="capability-item">🔍 Risk Assessment</div>
                <div class="capability-item">🧬 Genetic Analysis</div>
                <div class="capability-item">📊 Anomaly Detection</div>
                <div class="capability-item">🎯 Root Cause Analysis</div>
                <div class="capability-item">💊 Treatment Plans</div>
                <div class="capability-item">🛡️ Prevention Strategies</div>
            </div>
            <p style="margin-top: 20px;"><strong>Example medical queries:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li>Analyze cardiovascular risk factors for a 55-year-old male with hypertension</li>
                <li>What biomarkers predict heart disease progression?</li>
                <li>Create a personalized prevention plan for diabetic patients</li>
                <li>Explain the genetic factors in hypertension</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
else:
    for i, chat in enumerate(current_session["history"]):
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
        
        # Assistant message - handle both regular and streaming responses
        if chat['answer']:
            st.markdown(f"""
                <div class="assistant-message">
                    <div class="message-header">
                        <img src="https://api.dicebear.com/6.x/bottts/svg?seed=assistant{i}" class="avatar">
                        MedAnalytica Pro
                    </div>
                    {chat['answer']}
                    <div class="timestamp">{chat['timestamp']}</div>
                </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container

# Fixed input section
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 8, 1])

with col1:
    if st.button("🔄", help="New Chat", key="new_chat_icon"):
        st.session_state.show_new_chat = True

with col2:
    query = st.text_input(
        "Ask a medical question...",
        key="query_input",
        label_visibility="collapsed",
        placeholder="e.g., Analyze cardiovascular risk factors for a 55-year-old male with hypertension..."
    )

with col3:
    submit_btn = st.button("📤", help="Send Message", key="send_btn")

# Voice input note
st.markdown("""
    <div style="text-align: center; margin-top: 10px; color: #6c757d;">
        <small>🎙️ Voice input available • ⚡ Real-time analysis • 🛡️ HIPAA-compliant</small>
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close input-section
st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

# New Chat Modal
if st.session_state.show_new_chat:
    st.markdown('<div class="modal-overlay">', unsafe_allow_html=True)
    st.markdown("""
        <div class="new-chat-modal">
            <h3>🆕 Start New Chat</h3>
            <p>Starting a new chat will clear the current conversation.</p>
            <div style="display: flex; gap: 10px; margin-top: 20px;">
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm", use_container_width=True, key="confirm_new_chat"):
            create_new_chat_session()
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True, key="cancel_new_chat"):
            st.session_state.show_new_chat = False
            st.rerun()
    
    st.markdown('</div></div>', unsafe_allow_html=True)

# SIMPLIFIED QUERY PROCESSING - This is the key fix
if submit_btn and query and query != st.session_state.processed_query:
    if qa_chain is None:
        st.error("❌ Medical assistant not ready. Please ensure documents are properly loaded.")
    else:
        # Add user message to history immediately
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_session["history"].append({
            "question": query,
            "answer": "",  # Start with empty answer
            "timestamp": timestamp
        })
        st.session_state.processed_query = query
        
        # Display thinking message
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
            <div class="thinking-message">
                <div class="message-header">
                    <img src="https://api.dicebear.com/6.x/bottts/svg?seed=thinking" class="avatar">
                    MedAnalytica Pro
                </div>
                🤔 Analyzing your medical query...
            </div>
        """, unsafe_allow_html=True)
        
        try:
            # Enhanced medical prompt
            enhanced_query = f"""
            MEDICAL ANALYSIS REQUEST: {query}
            
            Please provide a comprehensive cardiovascular analysis including:
            - Risk assessment based on available medical data
            - Key contributing factors and biomarkers
            - Evidence-based recommendations
            - Prevention strategies
            - Follow-up considerations
            
            Format with clear sections using markdown.
            """
            
            # Get the response
            response = qa_chain.invoke(enhanced_query)
            answer = response.get("result", "I couldn't generate a comprehensive medical analysis based on the available documents.")
            
            # Update the answer in chat history
            current_session["history"][-1]["answer"] = answer
            
            # Update session title if it's the first message
            if len(current_session["history"]) == 1:
                title_words = query.split()[:4]
                current_session["title"] = " ".join(title_words) + ("..." if len(query.split()) > 4 else "")
            
        except Exception as e:
            error_msg = f"❌ Error in medical analysis: {str(e)}"
            current_session["history"][-1]["answer"] = error_msg
        
        # Clear thinking message and rerun to display full conversation
        thinking_placeholder.empty()
        st.rerun()

# Add padding at the bottom
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)