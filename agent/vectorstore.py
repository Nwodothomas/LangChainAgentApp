from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def build_vectorstore(docs, persist_path):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore

def load_vectorstore(persist_path):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
