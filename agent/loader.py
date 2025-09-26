import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from docx import Document as DocxReader

def load_documents(folder_path):
    docs = []

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(path)
            docs.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif filename.endswith(".docx"):
            doc = DocxReader(path)
            text = "\n".join([para.text for para in doc.paragraphs])
            docs.append(Document(page_content=text, metadata={"source": filename}))
    
    return docs