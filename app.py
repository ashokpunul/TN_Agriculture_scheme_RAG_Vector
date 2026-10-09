
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Configure the page
st.set_page_config(
    page_title="MultiPDF AI Assistant",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

st.title("📚 MultiPDF AI Assistant RAG")
st.caption("Ask questions and get answers based on your PDF documents.")

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Check your .env file.")
    st.stop()


# Load FAISS only once per Streamlit session cache
@st.cache_resource
def load_vector_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )


try:
    with st.spinner("Loading your PDF knowledge base..."):
        vector_db = load_vector_database()
        llm = load_llm()
except Exception as e:
    st.error(f"Could not load the RAG application: {e}")
    st.stop()


prompt = ChatPromptTemplate.from_template("""
Answer the user's question using the supplied PDF context.

If the context does not contain the answer, say that you
could not find the information in the provided PDFs.
Do not invent facts.

PDF context:
{context}

Question:
{question}

Answer:
""")


# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📄 View PDF sources"):
                for source in message["sources"]:
                    st.write(source)


# Chat input
question = st.chat_input("Ask a question about your PDFs...")

if question:
    # Display and save user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve relevant PDF chunks and generate answer
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching PDFs and generating an answer..."):
                documents = vector_db.similarity_search(
                    question,
                    k=3
                )

                if not documents:
                    answer = "I could not find relevant content in your PDFs."
                    sources = []
                else:
                    context = "\n\n".join(
                        doc.page_content for doc in documents
                    )

                    messages = prompt.format_messages(
                        context=context,
                        question=question
                    )

                    response = llm.invoke(messages)
                    answer = response.content

                    sources = [
                        f"File: {doc.metadata.get('source', 'Unknown')} | "
                        f"Page: {doc.metadata.get('page', 'Unknown')}"
                        for doc in documents
                    ]

            st.markdown(answer)

            if sources:
                with st.expander("📄 View PDF sources"):
                    for source in sources:
                        st.write(source)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as e:
            st.error(f"Something went wrong: {e}")