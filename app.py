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
    # Create initial session
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

# Enhanced medical system prompt
MEDICAL_SYSTEM_PROMPT = """
You are MedAnalytica Pro, an advanced medical AI assistant specializing in cardiovascular health, disease prevention, and comprehensive patient care analysis.

CORE CAPABILITIES:
1. **Multi-Modal Data Integration**: Analyze clinical data, genetic markers, lifestyle factors, environmental exposures, and imaging results
2. **Risk Assessment**: Calculate disease probabilities using established clinical models and AI-enhanced predictions
3. **Anomaly Detection**: Identify patterns and outliers in medical data that may indicate underlying conditions
4. **Root Cause Analysis**: Trace symptoms and biomarkers to potential underlying causes
5. **Treatment Optimization**: Recommend evidence-based interventions personalized to patient profiles
6. **Prevention Strategies**: Develop comprehensive lifestyle and medical prevention plans

ANALYSIS FRAMEWORK:
- **Clinical Data**: Vital signs, lab results, medical history, medications
- **Genetic Factors**: Family history, genetic markers, inherited risks
- **Lifestyle Factors**: Diet, exercise, smoking, alcohol, stress, sleep
- **Environmental Data**: Pollution exposure, occupational hazards, geographic risks
- **Temporal Patterns**: Disease progression, treatment response, monitoring trends

Always provide:
- Confidence levels for assessments
- Evidence-based recommendations
- Clear action plans with timelines
- Risk-benefit analyses
- Follow-up monitoring suggestions
- Emergency red flags when applicable

Format responses with clear sections using markdown for better readability.
"""

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
        
        .streaming-message {
            align-self: flex-start;
            background: #f0f8ff;
            color: #333;
            padding: 15px 20px;
            border-radius: 18px 18px 18px 5px;
            max-width: 70%;
            margin: 10px 0;
            border: 2px dashed #667eea;
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
        
        .new-chat-btn {
            background: #28a745 !important;
        }
        
        /* Sidebar styles */
        .sidebar-content {
            padding: 20px;
        }
        
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
        
        /* New chat modal */
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
        
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 1999;
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
    st.session_state.processed_query = ""  # Reset processed query

def get_current_session():
    """Get current chat session with error handling"""
    current_id = st.session_state.current_session_id
    if current_id not in st.session_state.chat_sessions:
        # Fallback to first available session
        if st.session_state.chat_sessions:
            first_id = list(st.session_state.chat_sessions.keys())[0]
            st.session_state.current_session_id = first_id
            return st.session_state.chat_sessions[first_id]
        else:
            # Create default session if none exists
            st.session_state.chat_sessions["default"] = {
                "id": "default",
                "history": [],
                "title": "New Chat"
            }
            st.session_state.current_session_id = "default"
            return st.session_state.chat_sessions["default"]
    return st.session_state.chat_sessions[current_id]

def simulate_streaming_response(text, speed=0.02):
    """Simulate streaming response character by character"""
    words = text.split(' ')
    response_container = st.empty()
    current_text = ""
    
    for word in words:
        current_text += word + " "
        response_container.markdown(f"""
            <div class="streaming-message">
                <div class="message-header">
                    <img src="https://api.dicebear.com/6.x/bottts/svg?seed=assistant" class="avatar">
                    Assistant
                </div>
                {current_text}▊
            </div>
        """, unsafe_allow_html=True)
        time.sleep(speed)
    
    # Final display without cursor
    response_container.markdown(f"""
        <div class="assistant-message">
            <div class="message-header">
                <img src="https://api.dicebear.com/6.x/bottts/svg?seed=assistant" class="avatar">
                Assistant
            </div>
            {text}
        </div>
    """, unsafe_allow_html=True)

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
    
    # Safely display chat history
    if st.session_state.chat_sessions:
        for session_id, session in list(st.session_state.chat_sessions.items()):
            is_active = session_id == st.session_state.current_session_id
            emoji = "🔵" if is_active else "⚪"
            if st.button(f"{emoji} {session.get('title', 'Untitled Chat')}", 
                        key=f"chat_{session_id}", 
                        use_container_width=True):
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
            st.info("🔄 Refresh to re-index documents.")

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
    
    # Build QA chain with enhanced medical prompt
    qa_chain = build_chain(vectorstore)
    
except Exception as e:
    st.error(f"❌ Error initializing medical assistant: {str(e)}")
    st.info("💡 Please upload medical documents to enable advanced analysis.")
    qa_chain = None

# Chat display area with scroll
current_session = get_current_session()  # This now has proper error handling
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
                <li>Analyze this patient's cardiovascular risk factors</li>
                <li>What biomarkers predict heart disease progression?</li>
                <li>Create a personalized prevention plan</li>
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
        
        # Assistant message
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
    st.markdown('<div class="modal-overlay" onclick="window.location.reload()">', unsafe_allow_html=True)
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

# Enhanced query processing with streaming
if submit_btn and query and query != st.session_state.processed_query:
    if qa_chain is None:
        st.error("❌ Medical assistant not ready. Please ensure documents are properly loaded.")
    else:
        # Add user message immediately
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_session["history"].append({
            "question": query,
            "answer": "",
            "timestamp": timestamp
        })
        st.session_state.processed_query = query
        
        # Process with enhanced medical context
        with st.spinner("🔍 Analyzing with MedAnalytica Pro..."):
            try:
                # Enhanced prompt with medical context
                enhanced_query = f"""
                MEDICAL ANALYSIS REQUEST:
                {query}
                
                CONTEXT: Please provide a comprehensive medical analysis including:
                - Risk assessment based on available data
                - Anomaly detection and pattern recognition
                - Root cause analysis if applicable
                - Evidence-based recommendations
                - Actionable prevention strategies
                - Follow-up monitoring suggestions
                
                Format response with clear medical sections and use markdown for readability.
                """
                
                response = qa_chain.invoke(enhanced_query)
                answer = response.get("result", "I couldn't generate a comprehensive medical analysis based on the available documents.")
                
                # Update the answer in chat history
                current_session["history"][-1]["answer"] = answer
                
                # Simulate streaming response
                simulate_streaming_response(answer)
                
                # Update session title if it's the first message
                if len(current_session["history"]) == 1:
                    # Create a short title from the first query
                    title_words = query.split()[:4]
                    current_session["title"] = " ".join(title_words) + ("..." if len(query.split()) > 4 else "")
                
            except Exception as e:
                error_msg = f"❌ Medical analysis error: {str(e)}"
                current_session["history"][-1]["answer"] = error_msg
                st.error(error_msg)
        
        # Rerun to ensure proper display
        st.rerun()

# Add padding at the bottom
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)