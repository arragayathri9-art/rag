import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langsmith import traceable

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

# ── Constants ─────────────────────────────────────────────────
CORPUS_PATH = os.getenv("CORPUS_PATH", "/kaggle/input/zyro-dynamics-hr-corpus/")

RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful HR assistant for Zyro Dynamics \
(also referred to as Acrux Dynamics in some documents).
Answer the employee's question using ONLY the information from the retrieved HR policy context below.
Be specific — include exact numbers, dates, durations, and policy rules when available.
Do not add any information not present in the context.

Retrieved HR Policy Context:
{context}

Employee Question: {question}

Answer:""")

OOS_PROMPT = ChatPromptTemplate.from_template("""You are a classifier. \
Decide if the following question is answerable from Zyro Dynamics internal HR policy documents.

HR documents cover: leave policies, WFH policy, compensation & benefits, performance reviews, \
onboarding & separation, POSH, IT & data security, travel & expense, code of conduct, and company profile.

Respond with exactly one word — IN_SCOPE or OUT_OF_SCOPE.

Question: {question}
Classification:""")

REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer questions about **Zyro Dynamics' internal HR policies** "
    "(leave, WFH, compensation, performance reviews, onboarding, code of conduct, etc.).\n\n"
    "Your question falls outside the scope of the available HR documents. "
    "Please contact the **HR team directly** or visit the company portal for further assistance."
)

SAMPLE_QUESTIONS = [
    "How many Earned Leave days do I get per year and how do they accrue?",
    "What is the maximum EL that can be carried forward at year end?",
    "How many weeks of maternity leave am I entitled to?",
    "What is required if I take sick leave for more than 2 consecutive days?",
    "By which date is my salary credited each month?",
    "What is the CTC range for an L4 Senior grade employee?",
    "What health insurance coverage does the company provide?",
    "When is an employee placed on a PIP and how long does it last?",
    "When does the Annual Performance Review happen and when are increments issued?",
    "Who is eligible for WFH and what types of WFH arrangements are available?",
]

# ── Pipeline (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner="📚 Loading HR policy documents — this may take a minute...")
def build_pipeline():
    loader    = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Use langchain_chroma (official package, Python 3.14 compatible)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="zyro_hr_docs"
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512
    )
    return retriever, llm

# ── Core logic ────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'HR Policy')}]\n{doc.page_content}"
        for doc in docs
    )

@traceable(name="streamlit-zyro-hr-bot")
def ask_bot(question: str, retriever, llm) -> tuple[str, list[str]]:
    scope_resp     = llm.invoke(OOS_PROMPT.format(question=question))
    classification = scope_resp.content.strip().upper()

    if "OUT_OF_SCOPE" in classification:
        return REFUSAL_MESSAGE, []

    docs    = retriever.invoke(question)
    context = format_docs(docs)
    resp    = llm.invoke(RAG_PROMPT.format(context=context, question=question))
    sources = sorted({
        os.path.basename(d.metadata.get("source", ""))
        for d in docs if d.metadata.get("source")
    })
    return resp.content, sources

# ── UI ────────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.markdown(
    "Ask any question about **Zyro Dynamics HR policies** — "
    "leave, WFH, compensation, performance reviews, onboarding, and more."
)
st.divider()

retriever, llm = build_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("💡 Sample Questions")
    st.markdown("Click any question to ask it:")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.pending_question = q

    st.divider()
    st.markdown("**Policies covered:**")
    for p in [
        "📋 Leave Policy", "🏠 Work From Home Policy", "💰 Compensation & Benefits",
        "📊 Performance Review", "🤝 Onboarding & Separation", "🛡️ POSH Policy",
        "💻 IT & Data Security", "✈️ Travel & Expense", "📖 Code of Conduct",
        "🏢 Company Profile", "📗 Employee Handbook",
    ]:
        st.caption(p)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Policy sources"):
                for s in msg["sources"]:
                    st.caption(f"• {s}")

# Handle sidebar button click
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")
else:
    user_input = st.chat_input("Ask an HR question...")

# Process input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            answer, sources = ask_bot(user_input, retriever, llm)
        st.markdown(answer)
        if sources:
            with st.expander("📄 Policy sources"):
                for s in sources:
                    st.caption(f"• {s}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.rerun()
