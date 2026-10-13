
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# Step 1: Define the folder containing our PDF files
PDF_FOLDER = Path("data")

# Step 2: Find all PDF files in the folder
pdf_files = list(PDF_FOLDER.glob("*.pdf"))

# Step 3: Check whether any PDFs were found
if not pdf_files:
    print("No PDF files found in the data folder.")
    print("Please add one or more PDF files to the data folder.")
else:
    all_documents = []

    # Step 4: Read each PDF file
    for pdf_file in pdf_files:
        print(f"\nLoading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()

        # Step 5: Add the extracted pages to our collection
        all_documents.extend(documents)

        print(f"Pages loaded: {len(documents)}")

    # Step 6: Display a summary
    print("\n--- Loading Summary ---")
    print(f"PDF files loaded: {len(pdf_files)}")
    print(f"Total pages loaded: {len(all_documents)}")

    # Step 7: Display text from the first page
    if all_documents:
        print("\n--- First Page Preview ---")
        print(all_documents[0].page_content[:1000])

        print("\n--- Page Metadata ---")
        print(all_documents[0].metadata)