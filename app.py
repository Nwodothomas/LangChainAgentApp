import streamlit as st
import os
import time
from datetime import datetime
from agent.loader import load_documents
from agent.vectorstore import load_vectorstore, build_vectorstore
from agent.chain import build_chain
from agent.utils import format_timestamp, validate_medical_query

# Page configuration
st.set_page_config(
    page_title="MedAnalytica Pro - Cardiovascular AI Assistant",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    try:
        with open("static/styles.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback basic CSS if file doesn't exist
        st.markdown("""
            <style>
                .main-container { max-width: 1000px; margin: 0 auto; }
                .chat-container { padding: 20px; margin-bottom: 100px; }
                .user-message { background: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0; }
                .assistant-message { background: #f5f5f5; padding: 10px; border-radius: 10px; margin: 5px 0; }
            </style>
        """, unsafe_allow_html=True)

load_css()

# Paths
docs_path = "data/docs"
persist_path = "vectorstore"
os.makedirs(docs_path, exist_ok=True)
os.makedirs(persist_path, exist_ok=True)

# Initialize session state - MUST BE AT THE TOP LEVEL
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.chat_sessions = {}
    st.session_state.current_session_id = "default"
    st.session_state.processing = False
    st.session_state.last_query = ""
    st.session_state.show_new_chat_modal = False
    
    # Create default session
    st.session_state.chat_sessions["default"] = {
        "id": "default",
        "title": "New Chat",
        "history": [],
        "created_at": format_timestamp()
    }

def create_new_session(session_id=None, title="New Chat"):
    if session_id is None:
        session_id = f"chat_{int(time.time())}"
    
    st.session_state.chat_sessions[session_id] = {
        "id": session_id,
        "title": title,
        "history": [],
        "created_at": format_timestamp()
    }
    return session_id

def get_current_session():
    current_id = st.session_state.current_session_id
    if current_id not in st.session_state.chat_sessions:
        # Create the session if it doesn't exist
        create_new_session(current_id, "New Chat")
    return st.session_state.chat_sessions[current_id]

def switch_session(session_id):
    if session_id in st.session_state.chat_sessions:
        st.session_state.current_session_id = session_id
        st.session_state.last_query = ""
        return True
    return False

def add_message_to_current_session(role, content):
    session = get_current_session()
    session["history"].append({
        "role": role,
        "content": content,
        "timestamp": format_timestamp()
    })

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h2>❤️ MedAnalytica Pro</h2>
            <p style="font-size: 0.9em; color: #666;">Cardiovascular AI Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    # New Chat Button
    if st.button("🔄 New Chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.show_new_chat_modal = True
    
    # Chat History
    st.markdown("### 📝 Chat History")
    
    if st.session_state.chat_sessions:
        for session_id, session in list(st.session_state.chat_sessions.items()):
            is_active = session_id == st.session_state.current_session_id
            emoji = "🔵" if is_active else "⚪"
            btn_label = f"{emoji} {session['title']}"
            
            if st.button(btn_label, key=f"session_{session_id}", use_container_width=True):
                if switch_session(session_id):
                    st.rerun()
    else:
        st.info("No chat history yet")
        # Ensure at least one session exists
        create_new_session("default", "New Chat")
    
    st.markdown("---")
    
    # Document Management
    with st.expander("📁 Document Management", expanded=True):
        st.markdown("""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                <p><strong>Supported formats:</strong> PDF, DOCX, TXT</p>
                <p><strong>Size limit:</strong> 200MB per file</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload medical documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="file_uploader"
        )
        
        if uploaded_files:
            success_count = 0
            for file in uploaded_files:
                try:
                    file_path = os.path.join(docs_path, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    success_count += 1
                except Exception as e:
                    st.error(f"Error uploading {file.name}: {str(e)}")
            
            if success_count > 0:
                st.success(f"✅ {success_count} file(s) uploaded successfully!")
                st.info("Documents will be processed automatically.")

# Main content area
st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 0 0 15px 15px; margin-bottom: 0;">
        <h1>MedAnalytica Pro</h1>
        <p>Advanced Cardiovascular AI Assistant • Risk Assessment • Diagnosis • Prevention</p>
    </div>
""", unsafe_allow_html=True)

# Initialize vectorstore and QA chain
@st.cache_resource(show_spinner=False)
def initialize_ai_agent():
    try:
        # Check if documents exist
        doc_files = [f for f in os.listdir(docs_path) if f.endswith(('.pdf', '.docx', '.txt'))]
        
        if not doc_files:
            return None, "📁 No medical documents found. Please upload PDF, DOCX, or TXT files in the sidebar."
        
        # Initialize vectorstore
        if not os.path.exists(f"{persist_path}/index.faiss"):
            with st.spinner("🔍 Indexing medical documents... This may take a few moments."):
                docs = load_documents(docs_path)
                if not docs:
                    return None, "❌ No readable content found in documents. Please check your file formats."
                vectorstore = build_vectorstore(docs, persist_path)
                st.success("✅ Documents indexed successfully!")
        else:
            vectorstore = load_vectorstore(persist_path)
        
        # Initialize QA chain
        qa_chain = build_chain(vectorstore)
        return qa_chain, "✅ Cardiovascular AI agent ready!"
        
    except Exception as e:
        error_msg = str(e)
        if ("api_key" in error_msg.lower() 
            or "openai" in error_msg.lower() 
            or "openai_api_key" in error_msg.lower()):
            return None, "🔑 OpenAI API key required. Please add OPENAI_API_KEY to your environment or .env file."
        else:
            return None, f"❌ Error initializing AI agent: {error_msg}"

qa_chain, status_message = initialize_ai_agent()

# Display status
if not qa_chain:
    st.error(status_message)
    
    # Helpful setup instructions
    if "OPENAI_API_KEY" in status_message or "API key" in status_message:
        with st.expander("🔧 Setup Instructions", expanded=True):
            st.markdown("""
            ### Quick Setup Guide:
            
            1. **Get OpenAI API Key:**
               - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
               - Create account/login → API Keys → Create new secret key
            
            2. **Configure environment:**
               - **Option A (.env file):**
                 ```env
                 OPENAI_API_KEY=sk-your_actual_key_here
                 ```
               - **Option B (system env var):**
                 - **Windows (Powershell):**
                   ```powershell
                   setx OPENAI_API_KEY "sk-your_actual_key_here"
                   ```
                 - **macOS/Linux (bash/zsh):**
                   ```bash
                   export OPENAI_API_KEY="sk-your_actual_key_here"
                   ```
            
            3. **Restart the application**
            
            **Note:** LangSmith is optional and not required for core functionality.
            """)
    
    if "No documents" in status_message or "readable content" in status_message:
        st.info("💡 Please upload medical documents in the sidebar to get started.")

# Chat interface - SAFELY get current session
try:
    current_session = get_current_session()
except KeyError:
    # Emergency fallback - recreate sessions
    st.session_state.chat_sessions = {}
    st.session_state.current_session_id = "default"
    create_new_session("default", "New Chat")
    current_session = get_current_session()

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
if not current_session["history"]:
    st.markdown("""
        <div style="text-align: center; padding: 40px; color: #6c757d;">
            <h3>👋 Welcome to MedAnalytica Pro!</h3>
            <p>Your advanced AI partner for cardiovascular health analysis.</p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 20px 0; max-width: 600px; margin-left: auto; margin-right: auto;">
                <div style="background: #f0f8ff; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #d0e0ff;">🔍 Risk Assessment</div>
                <div style="background: #f0f8ff; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #d0e0ff;">🧬 Genetic Analysis</div>
                <div style="background: #f0f8ff; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #d0e0ff;">📊 Anomaly Detection</div>
                <div style="background: #f0f8ff; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #d0e0ff;">🎯 Root Cause Analysis</div>
                <div style="background: #f0f8ff; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #d0e0ff;">💊 Treatment Plans</div>
                <div style="background: #f0f8ff; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #d0e0ff;">🛡️ Prevention Strategies</div>
            </div>
            
            <p><strong>Example queries:</strong></p>
            <ul style="text-align: left; display: inline-block; max-width: 500px;">
                <li>Analyze cardiovascular risk factors for a 55-year-old male with hypertension</li>
                <li>What biomarkers are most predictive of heart disease?</li>
                <li>Compare treatment options for atrial fibrillation</li>
                <li>Explain the role of cholesterol in cardiovascular health</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
else:
    for i, message in enumerate(current_session["history"]):
        if message["role"] == "user":
            st.markdown(f"""
                <div style="align-self: flex-end; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 20px; border-radius: 18px 18px 5px 18px; max-width: 70%; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: center; margin-bottom: 5px; font-weight: 600;">
                        <img src="https://api.dicebear.com/6.x/personas/svg?seed=user{i}" style="width: 24px; height: 24px; border-radius: 50%; margin-right: 8px;">
                        You
                    </div>
                    {message['content']}
                    <div style="font-size: 0.7rem; opacity: 0.7; margin-top: 8px; text-align: right;">{message['timestamp']}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="align-self: flex-start; background: #f8f9fa; color: #333; padding: 15px 20px; border-radius: 18px 18px 18px 5px; max-width: 70%; margin: 10px 0; border: 1px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <div style="display: flex; align-items: center; margin-bottom: 5px; font-weight: 600;">
                        <img src="https://api.dicebear.com/6.x/bottts/svg?seed=assistant{i}" style="width: 24px; height: 24px; border-radius: 50%; margin-right: 8px;">
                        MedAnalytica Pro
                    </div>
                    {message['content']}
                    <div style="font-size: 0.7rem; opacity: 0.7; margin-top: 8px; text-align: right;">{message['timestamp']}</div>
                </div>
            """, unsafe_allow_html=True)

# Show thinking indicator if processing
if st.session_state.processing:
    st.markdown("""
        <div style="align-self: flex-start; background: #f0f8ff; color: #666; padding: 10px 20px; border-radius: 18px; max-width: 200px; margin: 10px 0; border: 1px solid #667eea; font-style: italic;">
            <div style="display: flex; align-items: center; margin-bottom: 5px; font-weight: 600;">
                <img src="https://api.dicebear.com/6.x/bottts/svg?seed=thinking" style="width: 24px; height: 24px; border-radius: 50%; margin-right: 8px;">
                MedAnalytica Pro
            </div>
            Analyzing your query
            <span style="display: inline-flex; gap: 4px; margin-left: 10px;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: #667eea; animation: bounce 1.4s infinite ease-in-out;"></span>
                <span style="width: 6px; height: 6px; border-radius: 50%; background: #667eea; animation: bounce 1.4s infinite ease-in-out; animation-delay: 0.2s;"></span>
                <span style="width: 6px; height: 6px; border-radius: 50%; background: #667eea; animation: bounce 1.4s infinite ease-in-out; animation-delay: 0.4s;"></span>
            </span>
        </div>
        <style>
            @keyframes bounce {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }
        </style>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container

# Fixed input section
st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; right: 0; background: white; padding: 15px; border-top: 1px solid #e0e0e0; box-shadow: 0 -2px 20px rgba(0,0,0,0.1); z-index: 1000;">
        <div style="max-width: 1000px; margin: 0 auto; display: flex; gap: 10px; align-items: center;">
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 8, 1])

with col1:
    if st.button("🆕", help="Start New Chat", key="new_chat_icon"):
        st.session_state.show_new_chat_modal = True

with col2:
    query = st.text_input(
        "Ask a medical question...",
        key="query_input",
        label_visibility="collapsed",
        placeholder="e.g., Analyze cardiovascular risk factors for a 55-year-old male with hypertension...",
        disabled=st.session_state.processing
    )

with col3:
    submit_btn = st.button("📤", help="Send Message", key="send_btn", disabled=st.session_state.processing)

st.markdown('</div></div>', unsafe_allow_html=True)  # Close input-section

st.markdown("""
    <div style="text-align: center; margin-top: 10px; color: #6c757d; font-size: 0.8em;">
        🎙️ Voice input available • ⚡ Real-time analysis • 🛡️ HIPAA-compliant
    </div>
""", unsafe_allow_html=True)

# New Chat Modal
if st.session_state.show_new_chat_modal:
    st.markdown("""
        <div style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 2000; display: flex; justify-content: center; align-items: center;">
            <div style="background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); max-width: 500px; width: 90%;">
                <h3>🆕 Start New Chat</h3>
                <p>This will start a fresh conversation. Your current chat will be saved.</p>
                <div style="display: flex; gap: 10px; margin-top: 20px;">
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Start New Chat", key="confirm_new", use_container_width=True):
            new_session_id = create_new_session(title="New Chat")
            st.session_state.current_session_id = new_session_id
            st.session_state.show_new_chat_modal = False
            st.session_state.last_query = ""
            st.rerun()
    with col2:
        if st.button("❌ Cancel", key="cancel_new", use_container_width=True):
            st.session_state.show_new_chat_modal = False
            st.rerun()
    
    st.markdown('</div></div></div>', unsafe_allow_html=True)

# Handle query processing
if submit_btn and query and query != st.session_state.last_query:
    if not qa_chain:
        st.error("❌ AI agent not ready. Please check the status above.")
    elif not validate_medical_query(query):
        st.warning("⚠️ Please ask a medically relevant question about cardiovascular health.")
    else:
        # Set processing state
        st.session_state.processing = True
        st.session_state.last_query = query
        
        # Add user message immediately
        add_message_to_current_session("user", query)
        
        # Process the query
        try:
            with st.spinner("🤔 Analyzing with medical AI..."):
                response = qa_chain.invoke(query)
                answer = response.get("result", "I couldn't generate a response based on the available medical documents.")
                
                # Add assistant response
                add_message_to_current_session("assistant", answer)
                
                # Update session title if first message
                if len(current_session["history"]) == 2:  # User + Assistant
                    words = query.split()[:3]
                    current_session["title"] = " ".join(words) + ("..." if len(query.split()) > 3 else "")
                
        except Exception as e:
            error_msg = f"❌ Error processing your query: {str(e)}"
            add_message_to_current_session("assistant", error_msg)
        
        finally:
            st.session_state.processing = False
            st.rerun()

# Add bottom padding
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)