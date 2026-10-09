
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load the same embedding model used to create the database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load the saved FAISS database
vector_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded successfully!")


# 3. Ask a question
question = input("\nEnter your question: ")

# 4. Find the 3 most relevant PDF chunks
results = vector_db.similarity_search(
    question,
    k=3
)

# 5. Display the retrieved text
print("\nRelevant PDF chunks:")

for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print("Source:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))
    