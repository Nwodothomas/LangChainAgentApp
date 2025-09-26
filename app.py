import os
import time
from textwrap import dedent
import streamlit as st

from agent.loader import load_documents
from agent.vectorstore import load_vectorstore, build_vectorstore
from agent.chain import build_chain
from agent.utils import format_timestamp, validate_medical_query

# ---------- Page ----------
st.set_page_config(
    page_title="MedAnalytica Pro - Cardiovascular AI Assistant",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def render_html(html: str):
    st.markdown(dedent(html), unsafe_allow_html=True)

def load_css():
    try:
        with open("static/styles.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("<style>[data-testid='stAppViewContainer'] .main{overflow:hidden;}</style>", unsafe_allow_html=True)

load_css()

# ---------- Paths ----------
docs_path = "data/docs"
persist_path = "vectorstore"
os.makedirs(docs_path, exist_ok=True)
os.makedirs(persist_path, exist_ok=True)

# ---------- Session ----------
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.chat_sessions = {}
    st.session_state.current_session_id = "default"
    st.session_state.processing = False
    st.session_state.last_query = ""
    st.session_state.show_new_chat_modal = False
    st.session_state.chat_sessions["default"] = {
        "id": "default", "title": "New Chat", "history": [], "created_at": format_timestamp()
    }

def create_new_session(session_id=None, title="New Chat"):
    if session_id is None: session_id = f"chat_{int(time.time())}"
    st.session_state.chat_sessions[session_id] = {
        "id": session_id, "title": title, "history": [], "created_at": format_timestamp()
    }
    return session_id

def get_current_session():
    cid = st.session_state.current_session_id
    if cid not in st.session_state.chat_sessions:
        create_new_session(cid, "New Chat")
    return st.session_state.chat_sessions[cid]

def switch_session(session_id):
    if session_id in st.session_state.chat_sessions:
        st.session_state.current_session_id = session_id
        st.session_state.last_query = ""
        return True
    return False

def add_message_to_current_session(role, content):
    s = get_current_session()
    s["history"].append({"role": role, "content": content, "timestamp": format_timestamp()})

def count_uploaded_docs() -> int:
    try:
        return len([f for f in os.listdir(docs_path) if f.lower().endswith((".pdf",".docx",".txt"))])
    except Exception:
        return 0

# ---------- Left sidebar (unchanged) ----------
with st.sidebar:
    render_html("""
    <div class="sidebar-content" style="text-align:center; margin-bottom:30px;">
      <h2>❤️ MedAnalytica Pro</h2>
      <p style="font-size:0.9em; color:#666;">Cardiovascular AI Assistant</p>
    </div>
    """)
    if st.button("🔄 New Chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.show_new_chat_modal = True
    st.markdown("### 📝 Chat History")
    if st.session_state.chat_sessions:
        for session_id, session in list(st.session_state.chat_sessions.items()):
            is_active = session_id == st.session_state.current_session_id
            render_html(f'<div class="chat-history-item{" active" if is_active else ""}">')
            if st.button(f'{"🔵" if is_active else "⚪"} {session["title"]}',
                         key=f"session_{session_id}", use_container_width=True):
                if switch_session(session_id): st.rerun()
            render_html("</div>")
    else:
        st.info("No chat history yet"); create_new_session("default","New Chat")
    st.markdown("---")
    with st.expander("📁 Document Management", expanded=True):
        render_html('<div class="upload-section"><p><strong>Supported formats:</strong> PDF, DOCX, TXT</p><p><strong>Size limit:</strong> 200MB per file</p></div>')
        uploaded_files = st.file_uploader("Upload medical documents", type=["pdf","docx","txt"],
                                          accept_multiple_files=True, label_visibility="collapsed", key="file_uploader")
        if uploaded_files:
            ok = 0
            for f in uploaded_files:
                try:
                    with open(os.path.join(docs_path, f.name), "wb") as out: out.write(f.getbuffer()); ok += 1
                except Exception as e:
                    st.error(f"Error uploading {f.name}: {e}")
            if ok: st.success(f"✅ {ok} file(s) uploaded successfully!"); st.info("Documents will be processed automatically.")

# ---------- Center wrap + header ----------
render_html('<div class="center-wrap"><div class="main-wrap">')

# NEW: banner constrained to center width via .header-card
render_html("""
<div class="header-section">
  <div class="header-card">
    <h1>MedAnalytica Pro</h1>
    <p>Advanced Cardiovascular AI Assistant • Risk Assessment • Diagnosis • Prevention</p>
  </div>
</div>
""")

# ---------- Init QA chain ----------
@st.cache_resource(show_spinner=False)
def initialize_ai_agent():
    try:
        files = [f for f in os.listdir(docs_path) if f.endswith((".pdf",".docx",".txt"))]
        if not files:
            return None, "📁 No medical documents found. Please upload PDF, DOCX, or TXT files in the sidebar."
        if not os.path.exists(f"{persist_path}/index.faiss"):
            with st.spinner("🔍 Indexing medical documents... This may take a few moments."):
                docs = load_documents(docs_path)
                if not docs:
                    return None, "❌ No readable content found in documents. Please check your file formats."
                _ = build_vectorstore(docs, persist_path)
                st.success("✅ Documents indexed successfully!")
        vs = load_vectorstore(persist_path)
        return build_chain(vs), "✅ Cardiovascular AI agent ready!"
    except Exception as e:
        msg = str(e)
        if any(k in msg.lower() for k in ("api_key","openai","openai_api_key")):
            return None, "🔑 OpenAI API key required. Please add OPENAI_API_KEY to your environment or .env file."
        return None, f"❌ Error initializing AI agent: {msg}"

qa_chain, status_message = initialize_ai_agent()
if not qa_chain: st.error(status_message)

# ---------- Chat viewport (scrolls) ----------
render_html('<div class="chat-viewport"><div class="chat-container">')

try:
    current_session = get_current_session()
except KeyError:
    st.session_state.chat_sessions = {}; st.session_state.current_session_id = "default"
    create_new_session("default","New Chat"); current_session = get_current_session()

if not current_session["history"]:
    render_html('<div style="text-align:center; color:#98a2b3; padding:24px;">Start a conversation using the input below.</div>')
else:
    for i, message in enumerate(current_session["history"]):
        if message["role"] == "user":
            render_html(f"""
            <div class="user-message">
              <div class="message-header">
                <img src="https://api.dicebear.com/6.x/personas/svg?seed=user{i}" class="avatar"> You
              </div>
              {message['content']}
              <div class="timestamp">{message['timestamp']}</div>
            </div>""")
        else:
            render_html(f"""
            <div class="assistant-message">
              <div class="message-header">
                <img src="https://api.dicebear.com/6.x/bottts/svg?seed=assistant{i}" class="avatar"> MedAnalytica Pro
              </div>
              {message['content']}
              <div class="timestamp">{message['timestamp']}</div>
            </div>""")

if st.session_state.processing:
    render_html("""
    <div class="thinking-indicator">
      <div class="message-header">
        <img src="https://api.dicebear.com/6.x/bottts/svg?seed=thinking" class="avatar"> MedAnalytica Pro
      </div>
      Analyzing your query
      <span class="thinking-dots"><span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span></span>
    </div>""")

render_html("</div></div>")   # close chat-container + chat-viewport
render_html("</div></div>")   # close main-wrap + center-wrap

# ---------- Right sidebar (capabilities + info) ----------
def right_sidebar():
    docs_count = count_uploaded_docs()
    sess = get_current_session()
    render_html(f"""
    <aside class="right-pane">
      <div class="rp-section">
        <div class="rp-title">Capabilities</div>
        <div class="rp-card">
          <div class="rp-cap-grid">
            <div class="rp-cap-item">🔍 Risk Assessment</div>
            <div class="rp-cap-item">🧬 Genetic Analysis</div>
            <div class="rp-cap-item">📊 Anomaly Detection</div>
            <div class="rp-cap-item">🎯 Root Cause Analysis</div>
            <div class="rp-cap-item">💊 Treatment Plans</div>
            <div class="rp-cap-item">🛡️ Prevention Strategies</div>
          </div>
        </div>
      </div>

      <div class="rp-section">
        <div class="rp-title">Example queries</div>
        <div class="rp-card">
          <ul class="rp-list">
            <li>Analyze cardiovascular risk factors for a 55-year-old male with hypertension</li>
            <li>What biomarkers are most predictive of heart disease?</li>
            <li>Compare treatment options for atrial fibrillation</li>
            <li>Explain the role of cholesterol in cardiovascular health</li>
          </ul>
        </div>
      </div>

      <div class="rp-section">
        <div class="rp-title">Quick actions</div>
        <div class="rp-card">
          <span class="rp-chip">Summarize latest doc</span>
          <span class="rp-chip">List key biomarkers</span>
          <span class="rp-chip">Risk factors overview</span>
          <span class="rp-chip">AFib treatment compare</span>
        </div>
      </div>

      <div class="rp-section">
        <div class="rp-title">Documents</div>
        <div class="rp-card"><strong>{docs_count}</strong> uploaded (PDF/DOCX/TXT)</div>
      </div>

      <div class="rp-section">
        <div class="rp-title">Session</div>
        <div class="rp-card">
          <div><strong>Title:</strong> {sess['title']}</div>
          <div><strong>Created:</strong> {sess['created_at']}</div>
          <div><strong>Messages:</strong> {len(sess['history'])}</div>
        </div>
      </div>
    </aside>
    """)

right_sidebar()

# ---------- Input (fixed; never overlaps right pane) ----------
render_html('<div class="input-section"><div class="input-container">')
col1, col2, col3 = st.columns([1,8,1])
with col1:
    if st.button("🆕", help="Start New Chat", key="new_chat_icon"):
        st.session_state.show_new_chat_modal = True
with col2:
    query = st.text_input(
        "Ask a medical question...",
        key="query_input",
        label_visibility="collapsed",
        placeholder="e.g., Analyze cardiovascular risk factors for a 55-year-old male with hypertension...",
        disabled=st.session_state.processing,
    )
with col3:
    submit_btn = st.button("📤", help="Send Message", key="send_btn", disabled=st.session_state.processing)
render_html("</div></div>")

# ---------- Modal ----------
if st.session_state.show_new_chat_modal:
    render_html("""
    <div class="modal-overlay"><div class="modal-content">
      <h3>🆕 Start New Chat</h3><p>This will start a fresh conversation. Your current chat will be saved.</p>
      <div style="display:flex; gap:10px; margin-top:20px;">
    """)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Start New Chat", key="confirm_new", use_container_width=True):
            st.session_state.current_session_id = create_new_session(title="New Chat")
            st.session_state.show_new_chat_modal = False
            st.session_state.last_query = ""
            st.rerun()
    with c2:
        if st.button("❌ Cancel", key="cancel_new", use_container_width=True):
            st.session_state.show_new_chat_modal = False; st.rerun()
    render_html("</div></div></div>")

# ---------- Handle query ----------
if submit_btn and query and query != st.session_state.last_query:
    if not qa_chain:
        st.error("❌ AI agent not ready. Please check the status above.")
    elif not validate_medical_query(query):
        st.warning("⚠️ Please ask a medically relevant question about cardiovascular health.")
    else:
        st.session_state.processing = True
        st.session_state.last_query = query
        add_message_to_current_session("user", query)
        try:
            with st.spinner("🤔 Analyzing with medical AI..."):
                resp = qa_chain.invoke(query)
                answer = resp.get("result", "I couldn't generate a response based on the available medical documents.")
                add_message_to_current_session("assistant", answer)
                if len(get_current_session()["history"]) == 2:
                    words = query.split()[:3]
                    get_current_session()["title"] = " ".join(words) + ("..." if len(query.split()) > 3 else "")
        except Exception as e:
            add_message_to_current_session("assistant", f"❌ Error: {e}")
        finally:
            st.session_state.processing = False
            st.rerun()