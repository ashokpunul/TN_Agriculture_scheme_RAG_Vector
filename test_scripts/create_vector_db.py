
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# STEP 1: Find all PDF files
pdf_folder = Path("data")
pdf_files = list(pdf_folder.glob("*.pdf"))

if not pdf_files:
    print("No PDFs found in the data folder.")
    raise SystemExit(1)

# STEP 2: Load all PDF pages
all_documents = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()
    all_documents.extend(documents)
    print(f"Loaded {pdf_file.name}: {len(documents)} pages")

print(f"\nTotal pages: {len(all_documents)}")

# STEP 3: Split pages into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(all_documents)
print(f"Total chunks: {len(chunks)}")

# STEP 4: Load the embedding model
print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# STEP 5: Convert chunks to vectors and create FAISS index
print("Creating embeddings and FAISS database...")

vector_db = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# STEP 6: Save the database locally
vector_db.save_local("faiss_index")

print("\nSuccess! FAISS database created.")
print("Saved to: faiss_index/")
