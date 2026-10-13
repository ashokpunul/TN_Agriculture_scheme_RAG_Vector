## 🏗️ Project Architecture

MultiPDF AI Assistant is a Retrieval-Augmented Generation (RAG) application that retrieves relevant information from PDFs and web pages to generate context-aware answers.

### 🔄 Architecture Flow

```text
PDF Files / Web URLs
        ↓
Document Loader & Web Crawler
        ↓
Text Extraction, OCR & Image Processing
        ↓
Text Splitting (Chunking)
        ↓
Embeddings Generation
        ↓
FAISS Vector Database
        ↓
Semantic Search & Retrieval
        ↓
Relevant Context + User Question
        ↓
OpenAI LLM (GPT-4o-mini)
        ↓
Answer with Source References
        ↓
Streamlit User Interface
```

### 🧩 Main Components & Concepts

- **Streamlit UI:** Provides the interactive interface for uploading documents, entering URLs, asking questions, and viewing answers.
- **Document Loader:** Loads multiple PDFs and extracts text and metadata.
- **Web Crawler:** Visits web pages and follows relevant links to collect content.
- **OCR:** Extracts text from scanned PDF pages and images using Tesseract OCR.
- **Image Processing:** Detects and manages PDF images for relevant display or processing.
- **Text Splitter:** Divides large documents into smaller chunks for efficient retrieval.
- **Embeddings:** Converts text chunks into numerical vectors using a Hugging Face embedding model.
- **FAISS Vector Database:** Stores vectors and enables similarity-based retrieval.
- **Retriever:** Finds document chunks relevant to the user's question.
- **OpenAI LLM:** Uses the retrieved context to generate natural-language answers.
- **Source References:** Helps users identify the documents or web pages supporting an answer.
- **Incremental Indexing:** Updates indexed content when sources change, avoiding unnecessary reprocessing where supported.
- **Knowledge Base Reset:** Provides a controlled way to clear indexed data and rebuild the knowledge base.

### 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- OpenAI GPT-4o-mini
- Hugging Face Sentence Transformers
- FAISS
- PyMuPDF / PDF loaders
- Tesseract OCR
- BeautifulSoup / web crawling utilities

### 🎯 Key Concepts

- **RAG:** Combines document retrieval with LLM-generated answers.
- **Chunking:** Breaks documents into manageable text sections.
- **Embeddings:** Represents text as numerical vectors.
- **Vector Search:** Finds semantically similar content.
- **OCR:** Extracts text from scanned pages.
- **Web Crawling:** Collects information from URLs and linked pages.
- **Context-Aware Generation:** Grounds answers in retrieved information.
- **Incremental Processing:** Reduces repeated work when updating the knowledge base.

### 🚀 Benefits

- Search across multiple PDFs and web pages.
- Ask questions in natural language.
- Retrieve relevant information using semantic search.
- Process scanned documents using OCR.
- View relevant PDF images when requested.
- Trace answers to their source documents or web pages.
- Manage and rebuild the knowledge base."# TN_Agriculture_scheme_RAG_Vector" 
