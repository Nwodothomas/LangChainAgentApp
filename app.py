import streamlit as st
from agent.config import *
from agent.loader import load_documents
from agent.vectorstore import load_vectorstore, build_vectorstore
from agent.chain import build_chain
import os

st.set_page_config(page_title="Cardiovascular Study Assistant", layout="wide")
st.title("🧠 Cardiovascular Study Assistant")

persist_path = "vectorstore"
docs_path = "data/docs"

if not os.path.exists(f"{persist_path}/index.faiss"):
    with st.spinner("Loading and indexing documents..."):
        docs = load_documents(docs_path)
        vectorstore = build_vectorstore(docs, persist_path)
else:
    vectorstore = load_vectorstore(persist_path)

qa_chain = build_chain(vectorstore)

# Create a placeholder for the answer
answer_placeholder = st.empty()

# Input field BELOW the answer
query = st.text_input("Ask a question about the study:")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke(query)
        answer_placeholder.markdown(f"**Answer:** {response['result']}")