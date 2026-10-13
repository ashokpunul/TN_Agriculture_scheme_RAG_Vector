
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# STEP 1: Load the API key from .env
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not found. Check your .env file."
    )

print("Starting PDF RAG application...")

# STEP 2: Load the embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# STEP 3: Load the saved FAISS database
print("Loading FAISS database...")

vector_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded successfully!")

# STEP 4: Initialize OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# STEP 5: Create the RAG prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using the PDF context below.

If the answer is not available in the context,
say you could not find it in the PDFs.

PDF Context:
{context}

Question:
{question}

Answer:
""")

# STEP 6: Ask the user for a question
question = input("\nEnter your question: ").strip()

if not question:
    print("Please enter a question.")
    raise SystemExit(0)

# STEP 7: Retrieve relevant PDF chunks
print("\nSearching your PDFs...")

documents = vector_db.similarity_search(
    question,
    k=3
)

if not documents:
    print("No relevant PDF chunks found.")
    raise SystemExit(0)

context = "\n\n".join(
    doc.page_content for doc in documents
)

# STEP 8: Generate the AI answer
print("Generating AI answer...")

messages = prompt.format_messages(
    context=context,
    question=question
)

response = llm.invoke(messages)

# STEP 9: Display the answer
print("\n========== AI ANSWER ==========")
print(response.content)

# STEP 10: Display source documents
print("\n========== PDF SOURCES ==========")

for i, doc in enumerate(documents, start=1):
    print(f"\nSource {i}")
    print("File:", doc.metadata.get("source", "Unknown"))
    print("Page:", doc.metadata.get("page", "Unknown"))

print("\nRAG execution completed successfully!")