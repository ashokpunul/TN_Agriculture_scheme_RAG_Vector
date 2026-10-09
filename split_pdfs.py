
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# STEP 1: Locate the folder containing PDFs
PDF_FOLDER = Path("data")

# STEP 2: Find all PDF files
pdf_files = list(PDF_FOLDER.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found. Add PDFs to the data folder.")

else:
    # STEP 3: Load pages from every PDF
    all_documents = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        all_documents.extend(documents)

        print(f"Loaded: {pdf_file.name} ({len(documents)} pages)")

    # STEP 4: Configure the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # STEP 5: Split all loaded documents into chunks
    chunks = text_splitter.split_documents(all_documents)

    # STEP 6: Display the results
    print("\n--- SPLITTING SUMMARY ---")
    print(f"PDF files: {len(pdf_files)}")
    print(f"Pages loaded: {len(all_documents)}")
    print(f"Chunks created: {len(chunks)}")

    # STEP 7: Preview the first chunk
    if chunks:
        print("\n--- FIRST CHUNK ---")
        print(chunks[0].page_content)

        print("\n--- FIRST CHUNK METADATA ---")
        print(chunks[0].metadata)

        print("\n--- CHUNK LENGTH ---")
        print(len(chunks[0].page_content), "characters")