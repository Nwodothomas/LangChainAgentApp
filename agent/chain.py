# agent/chain.py
from typing import Any
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from agent.config import get_openai_api_key, OPENAI_MODEL, OPENAI_TEMPERATURE


CARDIO_PROMPT = """You are MedAnalytica Pro, a careful cardiovascular AI assistant.
Use ONLY the provided context to answer the user's question.
- If the context lacks an answer, say you cannot find it in the uploaded medical documents.
- Keep outputs evidence-driven, neutral, and avoid speculative claims.
- Do NOT provide medical diagnosis or treatment instructions; present information for educational purposes.
- Cite nothing; just answer concisely.

Context:
{context}

Question:
{question}

Answer:"""


def build_chain(vectorstore: Any):
    """
    Build a RetrievalQA chain over the provided vectorstore.
    vectorstore must implement .as_retriever().
    """
    api_key = get_openai_api_key()  # Raises a helpful error if missing

    llm = ChatOpenAI(
        api_key=api_key,
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=CARDIO_PROMPT,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

    return chain