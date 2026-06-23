import os
import glob
import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import BaseRetriever, Document
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langsmith import traceable
from pydantic import Field
from typing import List

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_PATH = os.environ.get("CORPUS_PATH", "/kaggle/input/zyro-dynamics-hr-corpus/")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

SAMPLE_QUESTIONS = [
    "How many earned leaves does an employee get per year?",
    "What is the carry-forward limit for earned leave?",
    "What is the maternity leave policy?",
    "How many sick leaves are allowed per year?",
    "On which date is salary credited each month?",
    "What is the CTC and bonus structure for L4 employees?",
    "What health insurance is provided to employees?",
    "What happens during a Performance Improvement Plan (PIP)?",
    "When does the Annual Performance Review (APR) take place?",
    "Who is eligible for Work From Home?",
]

OUT_OF_SCOPE_KEYWORDS = [
    "apply", "application", "recruitment", "hiring", "job opening",
    "esop", "stock option", "vesting",
    "revenue", "profit", "financial", "quarterly result",
    "acruxcrm", "salesforce", "competitor", "product feature",
    "zoho", "freshworks", "other company",
]

# ── Numpy Retriever ────────────────────────────────────────────────────────────
class NumpyRetriever(BaseRetriever):
    chunks: List[Document] = Field(default_factory=list)
    matrix: object = Field(default=None)
    embeddings_model: object = Field(default=None)
    k: int = Field(default=5)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        q_vec = np.array(
            self.embeddings_model.embed_query(query), dtype=np.float32
        )
        q_vec /= np.linalg.norm(q_vec) + 1e-10
        scores = self.matrix @ q_vec
        ranked = np.argsort(-scores)

        selected, seen_vecs = [], []
        for idx in ranked:
            if len(selected) >= self.k:
                break
            vec = self.matrix[idx]
            if any(np.dot(vec, s) > 0.95 for s in seen_vecs):
                continue
            selected.append(self.chunks[idx])
            seen_vecs.append(vec)
        return selected

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


# ── Pipeline ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading HR policy documents…")
def build_pipeline():
    # 1. Load PDFs
    pdf_files = glob.glob(os.path.join(CORPUS_PATH, "*.pdf"))
    if not pdf_files:
        st.error(f"No PDFs found at: {CORPUS_PATH}")
        st.stop()

    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # 3. Embed → numpy matrix
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    texts = [c.page_content for c in chunks]
    vecs = np.array(embeddings_model.embed_documents(texts), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    vecs /= norms

    retriever = NumpyRetriever(
        chunks=chunks, matrix=vecs, embeddings_model=embeddings_model, k=5
    )

    # 4. LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    return retriever, llm


# ── Guardrail ──────────────────────────────────────────────────────────────────
def is_out_of_scope(question: str) -> bool:
    q_lower = question.lower()
    return any(kw in q_lower for kw in OUT_OF_SCOPE_KEYWORDS)


# ── RAG Chain ──────────────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate.from_template("""
You are an HR assistant for Zyro Dynamics Pvt. Ltd.
Answer ONLY based on the context below. If the answer is not in the context, say
"I don't have that information in our HR policies."

Context:
{context}

Question: {question}

Answer:""")


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


@traceable(name="rag_chain")
def rag_chain(question: str, retriever, llm) -> dict:
    docs = retriever.get_relevant_documents(question)
    context = format_docs(docs)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer, "source_docs": docs}


@traceable(name="ask_bot")
def ask_bot(question: str, retriever, llm) -> dict:
    if is_out_of_scope(question):
        return {
            "answer": (
                "I'm only able to answer questions about Zyro Dynamics internal HR policies. "
                "This question appears to be outside that scope. Please contact HR directly "
                "for further assistance."
            ),
            "source_docs": [],
        }
    return rag_chain(question, retriever, llm)


# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢", layout="wide")
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask any question about Zyro Dynamics HR policies.")

# Sidebar
with st.sidebar:
    st.header("📋 Sample Questions")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q

    st.divider()
    st.header("📄 Policy Documents")
    policies = [
        "Company Profile", "Employee Handbook", "Leave Policy",
        "Work From Home Policy", "Code of Conduct",
        "Performance Review Policy", "Compensation & Benefits Policy",
        "IT & Data Security Policy", "POSH Policy",
        "Onboarding & Separation Policy", "Travel & Expense Policy",
    ]
    for p in policies:
        st.markdown(f"• {p}")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []

# Load pipeline
retriever, llm = build_pipeline()

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}** (page {src['page']})")
                    st.caption(src["snippet"])

# Handle sidebar button click
if "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")
else:
    prompt = st.chat_input("Ask an HR policy question…")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies…"):
            result = ask_bot(prompt, retriever, llm)
        answer = result["answer"]
        sources = [
            {
                "source": os.path.basename(d.metadata.get("source", "Unknown")),
                "page": d.metadata.get("page", "?"),
                "snippet": d.page_content[:200] + "…",
            }
            for d in result.get("source_docs", [])
        ]
        st.markdown(answer)
        if sources:
            with st.expander("📎 Sources"):
                for src in sources:
                    st.markdown(f"**{src['source']}** (page {src['page']})")
                    st.caption(src["snippet"])

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
