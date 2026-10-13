
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MultiPDF AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()


# =====================================================
# 2. CUSTOM PROFESSIONAL CSS
# =====================================================

st.markdown(
    """
    <style>
    /* Main page */
    .stApp {
        background: #f5f7fb;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.9);
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #172554 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #cbd5e1;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(120deg, #111827 0%, #312e81 65%, #4f46e5 100%);
        border-radius: 22px;
        padding: 32px;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 12px 35px rgba(49, 46, 129, 0.15);
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(28px, 4vw, 42px);
        font-weight: 750;
        letter-spacing: -1.2px;
        margin: 12px 0 10px 0;
        line-height: 1.15;
    }

    .hero p {
        color: #e0e7ff;
        font-size: 16px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .hero-label {
        display: inline-block;
        padding: 6px 12px;
        border: 1px solid rgba(255,255,255,.25);
        background: rgba(255,255,255,.10);
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: .5px;
        color: #e0e7ff;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e5eaf3;
        border-radius: 15px;
        padding: 18px;
        min-height: 112px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.025);
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 9px;
    }

    .metric-value {
        color: #0f172a;
        font-size: 23px;
        font-weight: 750;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e8edf5;
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.025);
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border-radius: 16px;
    }

    [data-testid="stChatInput"] textarea {
        font-size: 15px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
        transition: all .2s ease;
    }

    .stButton > button:hover {
        border-color: #6366f1;
        color: #4f46e5;
    }

    /* Section headings */
    .section-title {
        color: #0f172a;
        font-size: 21px;
        font-weight: 750;
        letter-spacing: -.4px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 18px;
    }

    /* Small footer */
    .app-footer {
        color: #94a3b8;
        text-align: center;
        font-size: 12px;
        padding: 25px 0 8px 0;
    }

    /* Status labels */
    .status-pill {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 12px;
        font-weight: 700;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 1rem 6rem 1rem;
        }

        .hero {
            padding: 23px;
            border-radius: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 3. SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 22px 0;">
            <div style="font-size: 34px;">📚</div>
            <div style="font-size: 22px; font-weight: 750;">
                MultiPDF AI
            </div>
            <div style="font-size: 12px; color: #a5b4fc;">
                INTELLIGENT DOCUMENT ASSISTANT
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### ✨ Workspace")
    st.caption("Ask questions across your indexed PDF documents.")

    st.markdown("### ⚙️ Configuration")
    st.markdown("- **LLM:** GPT-4o mini")
    st.markdown("- **Search:** FAISS similarity search")
    st.markdown("- **Embeddings:** MiniLM")
    st.markdown("- **Retrieval:** Top 3 chunks")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
        <div style="
            margin-top: 35px;
            padding: 14px;
            border: 1px solid #374151;
            border-radius: 12px;
            background: rgba(255,255,255,.04);
        ">
            <div style="font-weight: 700; margin-bottom: 5px;">
                🔒 Private workspace
            </div>
            <div style="font-size: 12px; color: #cbd5e1;">
                Your API key is read from your local environment.
                Keep secrets out of source control.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# 4. HERO HEADER
# =====================================================

st.markdown(
    """
    <div class="hero">
        <span class="hero-label">✦ AI-POWERED DOCUMENT SEARCH</span>
        <h1>Your PDFs.<br>One intelligent conversation.</h1>
        <p>
            Search your documents in natural language.
            Get contextual answers and inspect the PDF sources
            behind each response.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 5. CHECK API KEY
# =====================================================

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "OpenAI API key not found. Add OPENAI_API_KEY to your .env file."
    )
    st.stop()


# =====================================================
# 6. LOAD FAISS DATABASE AND LLM
# =====================================================

@st.cache_resource
def load_vector_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )


try:
    with st.spinner("Connecting to your PDF knowledge base..."):
        vector_db = load_vector_database()
        llm = load_llm()

except Exception as e:
    st.error("Unable to initialize the RAG application.")
    st.exception(e)
    st.stop()


# =====================================================
# 7. DASHBOARD METRICS
# =====================================================

index_exists = Path("faiss_index/index.faiss").exists()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">🧠 AI ENGINE</div>
            <div class="metric-value">GPT-4o mini</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">🔎 RETRIEVAL</div>
            <div class="metric-value">FAISS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    db_status = "Connected" if index_exists else "Unavailable"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">📚 KNOWLEDGE BASE</div>
            <div class="metric-value" style="font-size: 21px;">
                {db_status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


# =====================================================
# 8. RAG PROMPT
# =====================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a professional PDF research assistant.

Answer the user's question using the supplied PDF context.

Instructions:
- Use the provided context as your primary source.
- Do not invent facts or citations.
- If the context does not contain the answer, say that you
  could not find the information in the provided PDFs.
- Explain the answer clearly and organize it with bullets
  when that improves readability.

PDF context:
{context}

Question:
{question}

Answer:
"""
)


# =====================================================
# 9. CHAT HISTORY
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# Welcome panel shown when chat is empty
if not st.session_state.messages:
    st.markdown(
        '<div class="section-title">👋 Welcome to your workspace</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Start with a question or choose an example below.'
        '</div>',
        unsafe_allow_html=True,
    )

    suggestions = [
        ("📖 Explain a concept", "Explain the main concepts in my PDFs."),
        ("📝 Summarize documents", "Summarize the key points in my PDFs."),
        ("🎯 Find key details", "What are the most important details in my PDFs?"),
    ]

    suggestion_cols = st.columns(3)

    for i, (label, sample_question) in enumerate(suggestions):
        with suggestion_cols[i]:
            if st.button(label, use_container_width=True, key=f"suggest_{i}"):
                st.session_state.pending_question = sample_question
                st.rerun()


# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📄 View PDF sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")


# =====================================================
# 10. HANDLE USER QUESTION
# =====================================================

question = st.chat_input("Ask anything about your PDF documents...")

# Support suggested-question buttons
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and preparing your answer..."):

                documents = vector_db.similarity_search(
                    question,
                    k=3,
                )

                if not documents:
                    answer = (
                        "I could not find relevant content "
                        "in your indexed PDFs."
                    )
                    sources = []

                else:
                    context = "\n\n".join(
                        doc.page_content for doc in documents
                    )

                    messages = prompt.format_messages(
                        context=context,
                        question=question,
                    )

                    response = llm.invoke(messages)
                    answer = response.content

                    sources = []

                    for doc in documents:
                        filename = Path(
                            doc.metadata.get("source", "Unknown")
                        ).name

                        page = doc.metadata.get("page")

                        # PyPDFLoader page metadata is generally zero-based.
                        if isinstance(page, int):
                            page_display = page + 1
                        else:
                            page_display = "Unknown"

                        sources.append(
                            f"**{filename}** — Page {page_display}"
                        )

            st.markdown(answer)

            if sources:
                with st.expander("📄 View PDF sources"):
                    for source in sources:
                        st.markdown(f"- {source}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as e:
            st.error("Sorry, something went wrong while answering.")
            st.exception(e)


# =====================================================
# 11. FOOTER
# =====================================================

st.markdown(
    """
    <div class="app-footer">
        MultiPDF AI Assistant · Built with Streamlit, LangChain,
        FAISS & OpenAI
        <br>
        Answers are generated from retrieved document context.
        Verify important information against the original PDFs.
    </div>
    """,
    unsafe_allow_html=True,
)
