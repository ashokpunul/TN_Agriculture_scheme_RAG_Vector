
from urllib.parse import urlparse

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Configure embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Configure text splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

# 3. Extract content from a URL
def extract_url(url: str):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Enter a valid HTTP or HTTPS URL.")

    loader = WebBaseLoader(web_paths=(url,))
    documents = loader.load()

    if not documents or not any(
        doc.page_content.strip() for doc in documents
    ):
        raise ValueError("No readable text found at this URL.")

    for doc in documents:
        doc.metadata["source"] = url
        doc.metadata["source_type"] = "web"

    return documents


# 4. Split the extracted content
def split_documents(documents):
    return splitter.split_documents(documents)


# 5. Store in FAISS
def store_documents(chunks, index_path="faiss_index"):
    if not chunks:
        raise ValueError("No chunks available to index.")

    index_dir = __import__("pathlib").Path(index_path)

    if (index_dir / "index.faiss").exists():
        # Add to the existing index
        vector_db = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_db.add_documents(chunks)
    else:
        # Create a new index
        vector_db = FAISS.from_documents(
            chunks,
            embeddings,
        )

    vector_db.save_local(str(index_dir))
    return vector_db


if __name__ == "__main__":
    url = input("Enter webpage URL: ").strip()

    try:
        documents = extract_url(url)
        print("Extracted documents:", len(documents))

        chunks = split_documents(documents)
        print("Created chunks:", len(chunks))

        vector_db = store_documents(chunks)

        print("URL content saved to FAISS successfully.")

        # 6. Test retrieval
        question = input("Ask a question about the webpage: ")
        results = vector_db.similarity_search(question, k=3)

        for result in results:
            print("\nSource:", result.metadata.get("source"))
            print(result.page_content[:500])

    except Exception as exc:
        print(f"URL ingestion failed: {exc}")
