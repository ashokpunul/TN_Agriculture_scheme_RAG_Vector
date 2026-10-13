
import os
import io
import base64
import hashlib
import json
import re
import ipaddress
import socket
import csv
import tempfile
import shutil
from urllib.parse import urlparse, urljoin, urldefrag
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import streamlit as st
import pymupdf
import pytesseract
from PIL import Image
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Tesseract is installed at this Windows path on your computer.
# Using the full path avoids requiring a Windows PATH change.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Loader safety limits. Adjust these only if you understand the resource impact.
MAX_FILE_MB = 50
MAX_PAGES_PER_PDF = 2000
MAX_OCR_PAGES_PER_PDF = 30
MIN_TEXT_CHARS_BEFORE_OCR = 40
OCR_TIMEOUT_SECONDS = 30
DOCUMENT_STORE_DIR = Path("document_store")
TEXT_STORE_DIR = DOCUMENT_STORE_DIR / "extracted_text"
TABLES_DIR = DOCUMENT_STORE_DIR / "tables"
VISUALS_DIR = DOCUMENT_STORE_DIR / "page_previews"
EMBEDDED_IMAGES_DIR = DOCUMENT_STORE_DIR / "images"
MANIFEST_PATH = DOCUMENT_STORE_DIR / "manifest.json"
CHUNKS_CACHE_PATH = DOCUMENT_STORE_DIR / "chunks_cache.json"
MAX_EMBEDDED_IMAGES_PER_PAGE = 30
MAX_VISION_IMAGES_PER_PDF = 20
MAX_EMBEDDED_IMAGES_PER_PAGE = 30
MAX_PREVIEW_DIMENSION = 1400
MAX_PREVIEW_PIXELS = 4_000_000
MAX_WEBPAGE_BYTES = 2 * 1024 * 1024  # 2 MB per webpage
MAX_URLS_PER_PDF = 100
MAX_WEB_REDIRECTS = 0  # Redirects are deliberately disabled for SSRF safety


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MultiPDF AI Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()
if "messages" not in st.session_state:
    st.session_state.messages = []




def classify_answer_intent(question: str) -> str:
    """Classify the requested answer type so unrelated visuals are not displayed."""
    q = (question or "").lower()
    url_terms = ("url", "urls", "link", "links", "webpage", "web page", "website", "hyperlink")
    image_terms = ("image", "images", "picture", "pictures", "photo", "photos", "figure", "figures", "diagram", "diagrams")
    if any(term in q for term in url_terms):
        return "urls"
    if any(term in q for term in image_terms) or re.search(r"\\bshow\\s+(?:me\\s+)?image\\s*#?\\s*\\d+", q):
        return "images"
    return "text"


def is_url_count_request(question: str) -> bool:
    """Recognize questions asking how many URLs/links are present in PDFs."""
    q = (question or "").lower()
    asks_count = any(term in q for term in (
        "how many", "number of", "count", "total", "quantity",
    ))
    mentions_url = any(term in q for term in (
        "url", "urls", "link", "links", "hyperlink", "hyperlinks",
    ))
    mentions_pdf = any(term in q for term in ("pdf", "document", "file"))
    return asks_count and mentions_url and mentions_pdf


def filter_visuals_for_question(question: str, visuals: list) -> list:
    """Only show images when the user explicitly asks for image-related content."""
    if classify_answer_intent(question) != "images":
        return []
    q = (question or "").lower()
    match = re.search(r"\\bimage\\s*#?\\s*(\\d+)\\b", q)
    if match:
        wanted = int(match.group(1))
        for item in visuals:
            number = item.get("image_number", item.get("index"))
            try:
                if int(number) == wanted:
                    return [item]
            except (TypeError, ValueError):
                pass
        # Preserve existing behavior if this visual list is already pre-filtered.
        return visuals[:1]
    return visuals

def read_manifest():
    """Read the persistent document manifest, returning an empty manifest if absent/invalid."""
    if not MANIFEST_PATH.exists():
        return {"version": 1, "documents": {}}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("documents"), dict):
            raise ValueError("Unexpected manifest structure")
        return data
    except Exception as exc:
        st.warning(f"Document manifest could not be read; starting a new manifest. Details: {exc}")
        return {"version": 1, "documents": {}}


def write_manifest(manifest):
    """Atomically persist metadata so processed files can be reused across app restarts."""
    DOCUMENT_STORE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = MANIFEST_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(MANIFEST_PATH)


def manifest_assets(manifest):
    """Flatten persistent metadata into the lists used by the UI."""
    visual_pages, embedded_images, extracted_tables = [], [], []
    global_image_number = 0
    for filename, record in manifest.get("documents", {}).items():
        for visual in record.get("visual_pages", []):
            item = dict(visual)
            item.setdefault("source", filename)
            if Path(item.get("image_path", "")).is_file():
                visual_pages.append(item)
        for image in record.get("images", []):
            item = dict(image)
            item.setdefault("source", filename)
            if Path(item.get("image_path", "")).is_file():
                global_image_number += 1
                item["image_number"] = global_image_number
                embedded_images.append(item)
        for table in record.get("tables", []):
            item = dict(table)
            item.setdefault("source", filename)
            extracted_tables.append(item)
    return visual_pages, embedded_images, extracted_tables


def detect_heading(page):
    """Best-effort heading detection based on the largest readable text span on a page."""
    try:
        candidates = []
        page_dict = page.get_text("dict", sort=True)
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_value = " ".join(str(span.get("text", "")).split()).strip()
                    if text_value and len(text_value) <= 140:
                        candidates.append((float(span.get("size", 0)), text_value))
        if candidates:
            largest = max(size for size, _ in candidates)
            likely = [text_value for size, text_value in candidates if size >= largest * 0.94]
            return likely[0][:140] if likely else ""
    except Exception:
        pass
    return ""


def infer_heading_hierarchy(page_records):
    """Attach a best-effort section path across pages, inheriting the latest heading."""
    current_heading = ""
    for record in page_records:
        heading = str(record.get("heading", "") or "").strip()
        if heading:
            current_heading = heading
        record["section_path"] = record.get("section_path") or current_heading
    return page_records


def load_non_pdf_file(filename, file_bytes):
    """Load DOCX, TXT, Markdown, CSV, and Excel into normalized LangChain Documents."""
    suffix = Path(filename).suffix.lower()
    source_hash = hashlib.sha256(file_bytes).hexdigest()
    docs = []
    if suffix in {".txt", ".md", ".markdown"}:
        text_content = file_bytes.decode("utf-8", errors="replace")
        docs.append(Document(page_content=text_content, metadata={"source": filename, "source_type": "markdown" if suffix in {".md", ".markdown"} else "text", "document_id": source_hash[:16], "section_heading": ""}))
    elif suffix == ".csv":
        decoded = file_bytes.decode("utf-8-sig", errors="replace")
        rows = list(csv.reader(io.StringIO(decoded)))
        if rows:
            header = " | ".join(rows[0])
            # One logical row per line; headers stay attached for table-aware chunking.
            body = [" | ".join(row) for row in rows[1:]]
            content = "\n".join([header] + body)
            docs.append(Document(page_content=content, metadata={"source": filename, "source_type": "table", "file_format": "csv", "document_id": source_hash[:16], "table_header": header}))
    elif suffix == ".docx":
        from docx import Document as DocxDocument
        word_doc = DocxDocument(io.BytesIO(file_bytes))
        current_heading = []
        section_lines = []
        section_title = ""
        def flush_docx_section():
            if section_lines:
                docs.append(Document(
                    page_content="\n\n".join(section_lines),
                    metadata={"source": filename, "source_type": "docx", "document_id": source_hash[:16], "section_heading": section_title, "section_path": " > ".join(current_heading)},
                ))
        for para in word_doc.paragraphs:
            text_value = para.text.strip()
            if not text_value:
                continue
            style = (para.style.name or "") if para.style else ""
            if style.lower().startswith("heading"):
                flush_docx_section()
                section_lines = []
                try:
                    level = int(style.lower().replace("heading", "").strip())
                except Exception:
                    level = 1
                current_heading = current_heading[:max(0, level - 1)] + [text_value]
                section_title = text_value
            section_lines.append(text_value)
        flush_docx_section()
        for table_index, table in enumerate(word_doc.tables, start=1):
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            if rows:
                header = " | ".join(rows[0])
                content = "\n".join([header] + [" | ".join(row) for row in rows[1:]])
                docs.append(Document(page_content=content, metadata={"source": filename, "source_type": "table", "document_id": source_hash[:16], "table_number": table_index, "table_header": header, "section_path": " > ".join(current_heading)}))
    elif suffix in {".xlsx", ".xls"}:
        import pandas as pd
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, dtype=str)
        for sheet_name, frame in sheets.items():
            frame = frame.fillna("")
            if frame.empty:
                continue
            header = " | ".join(str(c) for c in frame.columns)
            rows = [header] + [header + "\n" + " | ".join(str(v) for v in row) for row in frame.itertuples(index=False, name=None)]
            docs.append(Document(page_content="\n".join(rows), metadata={"source": filename, "source_type": "table", "document_id": source_hash[:16], "sheet_name": sheet_name, "table_header": header}))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    if not docs or not any(d.page_content.strip() for d in docs):
        raise ValueError("No readable text or table data was found")
    return {"sha256": source_hash, "file_size_bytes": len(file_bytes), "source_type": suffix.lstrip("."), "documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs], "text_characters": sum(len(d.page_content) for d in docs)}


def passage_match(retrieved_docs, expected_passage):
    """Passage-level retrieval metrics using normalized expected passage matching."""
    normalize = lambda value: re.sub(r"\s+", " ", str(value).lower()).strip()
    expected = normalize(expected_passage)
    ranks = []
    for rank, doc in enumerate(retrieved_docs, start=1):
        content = normalize(doc.page_content)
        if expected and (expected in content or content in expected):
            ranks.append(rank)
    first_rank = min(ranks) if ranks else None
    return {"hit": bool(ranks), "reciprocal_rank": (1 / first_rank if first_rank else 0.0), "first_rank": first_rank}


def image_file_to_data_url(image_path, max_dimension=1400):
    """Resize/compress an image before sending it to a vision model."""
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        image.thumbnail((max_dimension, max_dimension))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def describe_image_with_vision(llm, image_path, source, page_number):
    """Create a grounded, searchable caption for an image using the vision-capable model."""
    data_url = image_file_to_data_url(image_path)
    message = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "Describe this PDF image for document search in 2-5 concise sentences. "
                "Include visible subject, legible labels, chart axes/values when readable, "
                "and the main takeaway if clearly supported. Do not guess unreadable details. "
                f"Source: {source}, page {page_number}."
            ),
        },
        {"type": "image_url", "image_url": {"url": data_url}},
    ])
    result = llm.invoke([message])
    return str(result.content).strip()


class LocalPDFFile:
    """Small adapter so local PDFs use the same processing pipeline as uploads."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = self.path.name

    def getvalue(self):
        return self.path.read_bytes()


def safe_source_name(name):
    """Prevent uploaded filenames from introducing path components."""
    return Path(str(name)).name


def extract_urls_from_page(page, page_text=""):
    """Collect clickable HTTP(S) links and URLs written in visible PDF text."""
    urls = set()
    try:
        for link in page.get_links():
            uri = link.get("uri")
            if isinstance(uri, str) and uri.lower().startswith(("http://", "https://")):
                urls.add(uri.strip())
    except Exception:
        pass
    for match in re.findall(r"https?://[^\s<>\"']+", page_text or "", flags=re.IGNORECASE):
        cleaned = match.rstrip(".,;:!?)]}>")
        if cleaned:
            urls.add(cleaned)
    return sorted(urls)


def validate_public_http_url(url):
    """Reject non-HTTP URLs and hostnames resolving to private/local IP addresses."""
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return False, "Only absolute HTTP/HTTPS URLs are supported."
        if parsed.username or parsed.password:
            return False, "URLs containing embedded credentials are not allowed."
        hostname = parsed.hostname.rstrip(".").lower()
        if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith((".local", ".internal")):
            return False, "Local/internal hostnames are blocked for safety."
        try:
            resolved = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        except OSError:
            return False, "The hostname could not be resolved."
        if not resolved:
            return False, "The hostname did not resolve to an address."
        for item in resolved:
            ip = ipaddress.ip_address(item[4][0].split("%", 1)[0])
            if not ip.is_global:
                return False, "Private, local, or reserved network addresses are blocked."
        return True, ""
    except Exception:
        return False, "The URL could not be validated."


def read_webpage_content(url):
    """Fetch a small HTML page safely; redirects and non-HTML resources are not followed."""
    valid, reason = validate_public_http_url(url)
    if not valid:
        return {"url": url, "title": url, "content": "", "error": reason}
    try:
        with requests.get(
            url,
            timeout=(5, 12),
            headers={"User-Agent": "MultiPDF-RAG-Reader/1.0"},
            allow_redirects=False,
            stream=True,
        ) as response:
            if 300 <= response.status_code < 400:
                return {"url": url, "title": url, "content": "", "error": "Redirects are blocked for safety. Open this URL manually if it redirects."}
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                return {"url": url, "title": url, "content": "", "error": f"Unsupported content type: {content_type or 'unknown'} (HTML only)."}
            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_WEBPAGE_BYTES:
                    return {"url": url, "title": url, "content": "", "error": "Webpage exceeds the 2 MB size limit."}
                chunks.append(chunk)
            raw = b"".join(chunks)
            encoding = response.encoding or "utf-8"
            html = raw.decode(encoding, errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else url
            # Collect navigable links before removing page chrome. The crawler later
            # restricts these to the same hostname and re-validates every URL.
            links = []
            for anchor in soup.find_all("a", href=True):
                href = (anchor.get("href") or "").strip()
                if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                    continue
                absolute = urldefrag(urljoin(url, href))[0]
                parsed_link = urlparse(absolute)
                if parsed_link.scheme.lower() in {"http", "https"} and parsed_link.hostname:
                    # Drop common non-page downloads and normalize the URL.
                    if not re.search(r"\.(pdf|png|jpe?g|gif|svg|webp|zip|docx?|xlsx?|pptx?|mp4|mp3)(?:$|\?)", parsed_link.path, re.I):
                        links.append(absolute)
            links = list(dict.fromkeys(links))
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
                tag.decompose()
            content = soup.get_text(" ", strip=True)
            content = re.sub(r"\s+", " ", content).strip()
            if not content:
                return {"url": url, "title": title, "content": "", "links": links, "error": "No readable text was found on this page."}
            return {"url": url, "title": title, "content": content, "links": links, "error": ""}
    except requests.RequestException as exc:
        return {"url": url, "title": url, "content": "", "error": f"Request failed: {type(exc).__name__}"}


def _split_semantically(document, chunk_size, chunk_overlap, embeddings):
    """Simple embedding-similarity sentence chunker; falls back to recursive splitting."""
    import re as _re
    text = document.page_content.strip()
    if not text:
        return []
    sentences = [part.strip() for part in _re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    if len(sentences) < 3:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        ).split_documents([document])
    try:
        vectors = embeddings.embed_documents(sentences)
        groups, current = [], [sentences[0]]
        current_len = len(sentences[0])
        previous = vectors[0]
        for sentence, vector in zip(sentences[1:], vectors[1:]):
            dot = sum(a * b for a, b in zip(previous, vector))
            norm_a = sum(a * a for a in previous) ** 0.5
            norm_b = sum(b * b for b in vector) ** 0.5
            similarity = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
            # Start a new chunk at a noticeable semantic change or size limit.
            if current_len + len(sentence) > chunk_size or similarity < 0.48:
                groups.append(" ".join(current))
                overlap_text = current[-1] if chunk_overlap > 0 else ""
                current = ([overlap_text] if overlap_text else []) + [sentence]
                current_len = sum(len(x) for x in current)
            else:
                current.append(sentence)
                current_len += len(sentence) + 1
            previous = vector
        if current:
            groups.append(" ".join(current))
        out = []
        search_from = 0
        for chunk_text in groups:
            metadata = dict(document.metadata)
            start_index = text.find(chunk_text[:80], search_from)
            metadata["start_index"] = max(start_index, 0)
            out.append(Document(page_content=chunk_text, metadata=metadata))
            if start_index >= 0:
                search_from = start_index + max(1, len(chunk_text) - chunk_overlap)
        return out
    except Exception:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        ).split_documents([document])


def split_documents_configurable(documents, settings):
    """Split loaded documents with selected strategy and table-aware row grouping."""
    chunk_size = int(settings.get("chunk_size", 1000))
    chunk_overlap = min(int(settings.get("chunk_overlap", 150)), max(0, chunk_size - 1))
    strategy = settings.get("splitter_strategy", "Recursive")
    embeddings = load_embeddings() if strategy == "Semantic" else None
    token_splitter = None
    if strategy == "Token":
        try:
            token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=int(settings.get("token_chunk_size", 350)),
                chunk_overlap=int(settings.get("token_overlap", 40)),
                add_start_index=True,
            )
        except Exception:
            # Optional tiktoken dependency: safely fall back to recursive character splitting.
            token_splitter = None
    output = []
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    for document in documents:
        # Table documents are prepared with repeated header rows; split at row boundaries first.
        if document.metadata.get("source_type") == "table":
            lines = document.page_content.splitlines()
            if len(lines) > 2:
                header = lines[0]
                rows = lines[1:]
                batch, batch_len = [], len(header)
                for row in rows:
                    if batch and batch_len + len(row) + 1 > chunk_size:
                        table_chunk = Document(
                            page_content=header + "\n" + "\n".join(batch),
                            metadata=dict(document.metadata),
                        )
                        output.extend(recursive.split_documents([table_chunk]))
                        batch, batch_len = [], len(header)
                    batch.append(row)
                    batch_len += len(row) + 1
                if batch:
                    table_chunk = Document(
                        page_content=header + "\n" + "\n".join(batch),
                        metadata=dict(document.metadata),
                    )
                    output.extend(recursive.split_documents([table_chunk]))
                continue
        if strategy == "Character":
            splitter = CharacterTextSplitter(
                separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                length_function=len, is_separator_regex=False,
            )
            output.extend(splitter.split_documents([document]))
        elif strategy == "Semantic":
            output.extend(_split_semantically(document, chunk_size, chunk_overlap, embeddings))
        elif strategy == "Token" and token_splitter is not None:
            output.extend(token_splitter.split_documents([document]))
        else:
            output.extend(recursive.split_documents([document]))
    # Optional parent-child mode: small child chunks retain their parent text and stable IDs.
    if settings.get("parent_child", False):
        parent_size = max(chunk_size, int(settings.get("parent_chunk_size", chunk_size * 2)))
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=chunk_overlap, add_start_index=True)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
        hierarchical = []
        for doc in documents:
            parents = parent_splitter.split_documents([doc])
            for pidx, parent in enumerate(parents):
                parent_id = hashlib.sha1((str(doc.metadata.get("source", "")) + str(doc.metadata.get("page_number", "")) + str(pidx) + parent.page_content[:100]).encode("utf-8")).hexdigest()[:16]
                for child in child_splitter.split_documents([parent]):
                    child.metadata.update(dict(parent.metadata))
                    child.metadata["parent_id"] = parent_id
                    child.metadata["parent_text"] = parent.page_content[:12000]
                    child.metadata["chunk_level"] = "child"
                    hierarchical.append(child)
        output = hierarchical
    for index, chunk in enumerate(output):
        chunk.metadata.setdefault("chunk_id", hashlib.sha1((str(chunk.metadata.get("source", "")) + str(chunk.metadata.get("page_number", "")) + str(index) + chunk.page_content).encode("utf-8")).hexdigest()[:16])
        chunk.metadata.setdefault("section_path", chunk.metadata.get("section_heading", ""))
    return output


def build_documents_from_manifest(manifest):
    """Convert cached PDF pages, extracted tables, and webpages into metadata-rich documents."""
    all_documents = []
    for source_name, record in manifest.get("documents", {}).items():
        for page_record in record.get("pages", []):
            content = str(page_record.get("page_text", "")).strip()
            if not content:
                continue
            page_number = int(page_record.get("page_number", 1))
            heading = str(page_record.get("heading", "") or "")
            if heading and not content.lower().startswith((heading.lower(), f"section/page heading: {heading.lower()}")):
                content = f"Section heading: {heading}\n\n{content}"
            all_documents.append(Document(
                page_content=content,
                metadata={
                    "source": source_name,
                    "document_id": record.get("sha256", "")[:16],
                    "page": page_number - 1,
                    "page_number": page_number,
                    "section_heading": heading,
                    "section_path": page_record.get("section_path", heading),
                    "extraction_method": page_record.get("extraction_method", "native text"),
                    "page_image": page_record.get("page_image", ""),
                    "source_type": "pdf",
                },
            ))
        # Keep tables separate and repeat the table header in every row group.
        for table in record.get("tables", []):
            rows = table.get("rows", [])
            if not rows:
                continue
            header = " | ".join(str(cell) for cell in rows[0])
            body_rows = [" | ".join(str(cell) for cell in row) for row in rows[1:]]
            table_text = "\n".join([header] + body_rows) if body_rows else header
            all_documents.append(Document(
                page_content=table_text,
                metadata={
                    "source": source_name,
                    "document_id": record.get("sha256", "")[:16],
                    "page": int(table.get("page_number", 1)) - 1,
                    "page_number": int(table.get("page_number", 1)),
                    "section_heading": "",
                    "table_number": table.get("table_number", 1),
                    "source_type": "table",
                },
            ))
    for source_name, record in manifest.get("other_documents", {}).items():
        for item in record.get("documents", []):
            content = str(item.get("page_content", "")).strip()
            if content:
                metadata = dict(item.get("metadata", {}))
                metadata.update({"source": source_name, "document_id": record.get("sha256", "")[:16], "source_type": metadata.get("source_type", record.get("source_type", "document"))})
                all_documents.append(Document(page_content=content, metadata=metadata))
    for url, record in manifest.get("webpages", {}).items():
        content = str(record.get("content", "")).strip()
        if content:
            all_documents.append(Document(
                page_content=f"Webpage title: {record.get('title', url)}\nURL: {url}\n\n{content}",
                metadata={"source": url, "url": url, "title": record.get("title", url), "source_type": "webpage"},
            ))
    return all_documents


def preview_chunks(manifest, settings, limit=100):
    """Return metadata-rich chunks for preview and indexing."""
    docs = build_documents_from_manifest(manifest)
    chunks = split_documents_configurable(docs, settings)
    return chunks[:limit], len(chunks)


def make_chunk_cache_key(manifest, settings):
    """Stable cache key; any changed or removed source invalidates cached chunks."""
    cache_payload = {
        "settings": settings,
        "documents": {name: record.get("sha256", "") for name, record in manifest.get("documents", {}).items()},
        "webpages": {url: hashlib.sha256(str(rec.get("content", "")).encode("utf-8")).hexdigest() for url, rec in manifest.get("webpages", {}).items()},
        "other_documents": {name: record.get("sha256", "") for name, record in manifest.get("other_documents", {}).items()},
    }
    return hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()


def chunk_cache_matches(cache_data, manifest, settings):
    """Return True only when cached chunks match every current source and setting."""
    return bool(isinstance(cache_data, dict) and cache_data.get("cache_key") == make_chunk_cache_key(manifest, settings))


def validate_index_sources(manifest, indexed_sources):
    """Compare indexed source metadata with manifest sources to detect stale entries."""
    expected = set(manifest.get("documents", {})) | set(manifest.get("other_documents", {})) | set(manifest.get("webpages", {}))
    indexed = {str(source) for source in indexed_sources if str(source)}
    return {"stale_sources": sorted(indexed - expected), "missing_sources": sorted(expected - indexed)}


def _source_hashes_from_manifest(manifest):
    """Return stable content hashes for every PDF, non-PDF document, and webpage."""
    hashes = {}
    for name, record in manifest.get("documents", {}).items():
        hashes[str(name)] = str(record.get("sha256") or "")
    for name, record in manifest.get("other_documents", {}).items():
        hashes[str(name)] = str(record.get("sha256") or "")
    for url, record in manifest.get("webpages", {}).items():
        # Include title as it is included in the indexed page content.
        payload = json.dumps(
            {"content": str(record.get("content", "")), "title": str(record.get("title", url))},
            sort_keys=True,
            ensure_ascii=False,
        )
        hashes[str(url)] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return hashes


def _settings_fingerprint(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _source_chunk_cache_path(source):
    safe_key = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    return DOCUMENT_STORE_DIR / "source_chunk_cache" / f"{safe_key}.json"


def _read_source_chunk_cache(source, source_hash, settings_hash):
    path = _source_chunk_cache_path(source)
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (cached.get("source") == source
                and cached.get("source_hash") == source_hash
                and cached.get("settings_hash") == settings_hash
                and isinstance(cached.get("chunks"), list)):
            return [Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
                    for item in cached["chunks"]]
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def _write_source_chunk_cache(source, source_hash, settings_hash, chunks):
    path = _source_chunk_cache_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "source_hash": source_hash,
        "settings_hash": settings_hash,
        "chunks": [{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _ordered_index_documents(vector_db):
    """Read indexed documents in FAISS vector order for previews and compatibility cache."""
    ordered = []
    mapping = getattr(vector_db, "index_to_docstore_id", {}) or {}
    for vector_position, docstore_id in sorted(mapping.items(), key=lambda item: item[0]):
        try:
            doc = vector_db.docstore.search(docstore_id)
            if isinstance(doc, Document):
                ordered.append(doc)
        except Exception:
            continue
    return ordered


def _vector_ids_for_sources(vector_db, sources):
    source_set = set(sources)
    ids = []
    for docstore_id in (getattr(vector_db, "index_to_docstore_id", {}) or {}).values():
        try:
            doc = vector_db.docstore.search(docstore_id)
            if isinstance(doc, Document) and str(doc.metadata.get("source", "")) in source_set:
                ids.append(docstore_id)
        except Exception:
            continue
    return ids


def _save_faiss_atomic(vector_db, target="faiss_index_uploaded"):
    """Save a FAISS snapshot and swap it into place with Windows-safe retries.

    Windows can temporarily deny directory renames when another Streamlit rerun,
    antivirus scanner, or file indexer is touching the FAISS files. Keep a backup
    until the replacement succeeds, and use absolute paths to avoid working-dir
    ambiguity.
    """
    import os
    import shutil
    import time
    import uuid

    target_path = Path(target).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(
        prefix=f".{target_path.name}_tmp_", dir=str(target_path.parent)
    )).resolve()
    backup_path = target_path.with_name(f".{target_path.name}_backup_{uuid.uuid4().hex[:8]}")
    moved_old_index = False

    def rename_with_retry(source, destination, attempts=8):
        last_error = None
        for attempt in range(attempts):
            try:
                source.rename(destination)
                return
            except PermissionError as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(0.25 * (attempt + 1))
        raise last_error

    try:
        vector_db.save_local(str(temp_path))
        # Confirm both files exist before touching the currently usable index.
        if not (temp_path / "index.faiss").is_file() and not (temp_path / "index.faiss").exists():
            raise RuntimeError("FAISS save did not create index.faiss.")
        if not (temp_path / "index.pkl").is_file():
            raise RuntimeError("FAISS save did not create index.pkl.")

        if target_path.exists():
            rename_with_retry(target_path, backup_path)
            moved_old_index = True

        try:
            rename_with_retry(temp_path, target_path)
        except Exception:
            # Restore the old index if installing the new snapshot fails.
            if moved_old_index and backup_path.exists() and not target_path.exists():
                rename_with_retry(backup_path, target_path)
                moved_old_index = False
            raise

        if moved_old_index and backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=True)
            moved_old_index = False

    except PermissionError as exc:
        raise PermissionError(
            f"Windows denied access while replacing '{target_path}'. "
            "Stop duplicate Streamlit instances using this project (Ctrl+C in each terminal), "
            "close File Explorer windows previewing the index folder, then retry. "
            f"Original error: {exc}"
        ) from exc
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        # Never delete a backup if the replacement failed and it may be the only good index.


def rebuild_uploaded_index(manifest, settings=None):
    """Incrementally update the managed FAISS index for only new/changed/deleted sources.

    PDF/OCR/image/table extraction and other-format/webpage loading remain upstream in the
    manifest. This function groups their normalized documents by source, reuses per-source
    chunk caches, embeds only changed/new chunks, and removes stale vectors before saving.
    """
    settings = settings or {"chunk_size": 1000, "chunk_overlap": 150, "splitter_strategy": "Recursive"}
    all_documents = build_documents_from_manifest(manifest)
    managed_index_path = Path("faiss_index_uploaded")
    index_files_exist = (managed_index_path / "index.faiss").exists() and (managed_index_path / "index.pkl").exists()

    # No source documents means the managed uploaded index should be empty. The original
    # user's separate `faiss_index` is deliberately never modified.
    if not all_documents:
        import shutil
        if managed_index_path.exists():
            shutil.rmtree(managed_index_path, ignore_errors=True)
        try:
            CHUNKS_CACHE_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        cache_dir = DOCUMENT_STORE_DIR / "source_chunk_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        return None, 0, []

    source_hashes = _source_hashes_from_manifest(manifest)
    settings_hash = _settings_fingerprint(settings)
    grouped_documents = {}
    for doc in all_documents:
        source = str(doc.metadata.get("source", "") or "Unknown source")
        grouped_documents.setdefault(source, []).append(doc)
    current_sources = set(grouped_documents)

    vector_db = None
    if index_files_exist:
        try:
            vector_db = FAISS.load_local(
                str(managed_index_path), load_embeddings(), allow_dangerous_deserialization=True
            )
        except Exception:
            # Corrupt/old index: rebuild safely from the source caches below.
            vector_db = None

    indexed_docs = _ordered_index_documents(vector_db) if vector_db is not None else []
    indexed_sources = {str(doc.metadata.get("source", "")) for doc in indexed_docs}
    index_needs_full_rebuild = vector_db is None

    # Legacy indexes are usable: their vectors are removed source-by-source and replaced as
    # needed. Missing cache metadata merely causes a source to be chunked again, not data loss.
    changed_sources = set()
    cached_chunks_by_source = {}
    for source, source_docs in grouped_documents.items():
        source_hash = source_hashes.get(source, "")
        cached_chunks = _read_source_chunk_cache(source, source_hash, settings_hash)
        if (vector_db is not None and source in indexed_sources and cached_chunks is not None):
            cached_chunks_by_source[source] = cached_chunks
            continue
        if cached_chunks is not None and vector_db is not None and source not in indexed_sources:
            # A valid chunk cache can repopulate a source missing from the index without re-splitting.
            cached_chunks_by_source[source] = cached_chunks
            changed_sources.add(source)
            continue
        chunks = split_documents_configurable(source_docs, settings)
        for chunk_number, chunk in enumerate(chunks):
            chunk.metadata["source"] = source
            chunk.metadata["source_hash"] = source_hash
            chunk.metadata["chunk_id"] = hashlib.sha256(
                f"{source}\0{settings_hash}\0{chunk_number}\0{chunk.page_content}".encode("utf-8")
            ).hexdigest()[:24]
        cached_chunks_by_source[source] = chunks
        _write_source_chunk_cache(source, source_hash, settings_hash, chunks)
        changed_sources.add(source)

    stale_sources = indexed_sources - current_sources
    sources_to_remove = stale_sources | (changed_sources & indexed_sources)

    # If the index could not be loaded, build a fresh one from all sources. Otherwise mutate
    # only the affected source IDs in memory and persist atomically after all operations succeed.
    if index_needs_full_rebuild:
        chunks_to_add = []
        for source in sorted(current_sources):
            chunks = cached_chunks_by_source[source]
            if not chunks:
                continue
            for chunk_number, chunk in enumerate(chunks):
                chunk.metadata["source"] = source
                chunk.metadata["source_hash"] = source_hashes.get(source, "")
                chunk.metadata.setdefault("chunk_id", hashlib.sha256(
                    f"{source}\0{settings_hash}\0{chunk_number}\0{chunk.page_content}".encode("utf-8")
                ).hexdigest()[:24])
            chunks_to_add.extend(chunks)
        if not chunks_to_add:
            return None, 0, []
        ids = [hashlib.sha256(
            f"{doc.metadata.get('source', '')}\0{doc.metadata.get('chunk_id', '')}".encode("utf-8")
        ).hexdigest() for doc in chunks_to_add]
        vector_db = FAISS.from_documents(chunks_to_add, load_embeddings(), ids=ids)
    else:
        if sources_to_remove:
            old_ids = _vector_ids_for_sources(vector_db, sources_to_remove)
            if old_ids:
                try:
                    vector_db.delete(ids=old_ids)
                except Exception as exc:
                    # A targeted delete failure must not leave stale vectors. Fall back to a
                    # full, clean rebuild from cached chunks rather than saving a mixed index.
                    chunks_to_add = [chunk for source in sorted(current_sources)
                                     for chunk in cached_chunks_by_source[source]]
                    if not chunks_to_add:
                        return None, 0, []
                    ids = [hashlib.sha256(
                        f"{doc.metadata.get('source', '')}\0{doc.metadata.get('chunk_id', '')}".encode("utf-8")
                    ).hexdigest() for doc in chunks_to_add]
                    vector_db = FAISS.from_documents(chunks_to_add, load_embeddings(), ids=ids)
                    sources_to_remove = set()
                    changed_sources = set()  # The fallback already indexed every current source.

        chunks_to_add = [chunk for source in sorted(changed_sources)
                         for chunk in cached_chunks_by_source.get(source, [])]
        if chunks_to_add:
            ids = [hashlib.sha256(
                f"{doc.metadata.get('source', '')}\0{doc.metadata.get('chunk_id', '')}".encode("utf-8")
            ).hexdigest() for doc in chunks_to_add]
            try:
                vector_db.add_documents(chunks_to_add, ids=ids)
            except Exception:
                # Do not save a partially updated index. Rebuild from the per-source caches.
                all_cached_chunks = [chunk for source in sorted(current_sources)
                                     for chunk in cached_chunks_by_source[source]]
                if not all_cached_chunks:
                    return None, 0, []
                all_ids = [hashlib.sha256(
                    f"{doc.metadata.get('source', '')}\0{doc.metadata.get('chunk_id', '')}".encode("utf-8")
                ).hexdigest() for doc in all_cached_chunks]
                vector_db = FAISS.from_documents(all_cached_chunks, load_embeddings(), ids=all_ids)

    _save_faiss_atomic(vector_db, str(managed_index_path))

    # Keep the legacy aggregate cache for the existing integrity UI and backwards compatibility.
    final_chunks = _ordered_index_documents(vector_db)
    cache_key_hash = make_chunk_cache_key(manifest, settings)
    try:
        CHUNKS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_cache = CHUNKS_CACHE_PATH.with_suffix(".tmp")
        temp_cache.write_text(json.dumps({
            "cache_key": cache_key_hash,
            "settings": settings,
            "chunks": [{"page_content": c.page_content, "metadata": c.metadata} for c in final_chunks],
            "incremental": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_cache.replace(CHUNKS_CACHE_PATH)
    except Exception:
        pass

    # Prune per-source chunk caches for sources that are no longer present.
    cache_dir = DOCUMENT_STORE_DIR / "source_chunk_cache"
    if cache_dir.exists():
        live_cache_paths = {_source_chunk_cache_path(source).resolve() for source in current_sources}
        for cache_path in cache_dir.glob("*.json"):
            if cache_path.resolve() not in live_cache_paths:
                try:
                    cache_path.unlink()
                except OSError:
                    pass

    indexed_sources_after = {str(doc.metadata.get("source", "")) for doc in final_chunks}
    # A manifest entry can legitimately contain only images/empty pages and therefore produce
    # no text chunks. Validate against sources that yielded normalized text/table documents.
    indexable_manifest = {
        "documents": {k: v for k, v in manifest.get("documents", {}).items() if k in current_sources},
        "other_documents": {k: v for k, v in manifest.get("other_documents", {}).items() if k in current_sources},
        "webpages": {k: v for k, v in manifest.get("webpages", {}).items() if k in current_sources},
    }
    consistency = validate_index_sources(indexable_manifest, indexed_sources_after)
    if consistency["stale_sources"] or consistency["missing_sources"]:
        raise RuntimeError(
            "Incremental FAISS update failed source integrity validation: "
            f"stale={consistency['stale_sources'][:5]}, missing={consistency['missing_sources'][:5]}"
        )
    return vector_db, len(final_chunks), final_chunks


# =====================================================


def _rag_tokens(text):
    """Normalize English and numeric terms for lightweight BM25 keyword retrieval."""
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", str(text).lower())


def bm25_search(question, documents, k=30, k1=1.5, b=0.75):
    """Small dependency-free BM25 implementation over the current indexed chunks."""
    if not documents:
        return []
    query_terms = _rag_tokens(question)
    tokenized = [_rag_tokens(doc.page_content) for doc in documents]
    if not query_terms or not any(tokenized):
        return []
    lengths = [len(tokens) for tokens in tokenized]
    avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0
    doc_freq = {}
    term_freqs = []
    for tokens in tokenized:
        counts = {}
        for term in tokens:
            counts[term] = counts.get(term, 0) + 1
        term_freqs.append(counts)
        for term in counts:
            doc_freq[term] = doc_freq.get(term, 0) + 1
    n_docs = len(documents)
    scores = []
    for idx, counts in enumerate(term_freqs):
        score = 0.0
        dl = lengths[idx]
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = doc_freq.get(term, 0)
            idf = max(0.0, __import__("math").log(1 + (n_docs - df + 0.5) / (df + 0.5)))
            denom = tf + k1 * (1 - b + b * (dl / avgdl if avgdl else 0))
            score += idf * (tf * (k1 + 1) / denom if denom else 0)
        if score > 0:
            scores.append((score, idx))
    scores.sort(reverse=True)
    return [documents[idx] for _, idx in scores[:max(1, int(k))]]


def _doc_identity(doc):
    """Stable identity used to merge retrieval result lists."""
    meta = doc.metadata or {}
    return str(meta.get("chunk_id") or hashlib.sha1(
        (str(meta.get("source", "")) + "|" + str(meta.get("page_number", "")) + "|" + doc.page_content).encode("utf-8")
    ).hexdigest())


def hybrid_retrieve(question, vector_db, k=30, use_hybrid=True, rrf_constant=60):
    """Combine FAISS semantic results with BM25 keyword results using RRF."""
    if vector_db is None:
        return []
    vector_docs = vector_db.similarity_search(question, k=k)
    if not use_hybrid:
        return vector_docs
    corpus = _ordered_index_documents(vector_db)
    keyword_docs = bm25_search(question, corpus, k=k)
    fused = {}
    for rank, doc in enumerate(vector_docs, start=1):
        key = _doc_identity(doc)
        fused.setdefault(key, {"doc": doc, "score": 0.0, "channels": set()})
        fused[key]["score"] += 1.0 / (rrf_constant + rank)
        fused[key]["channels"].add("FAISS")
    for rank, doc in enumerate(keyword_docs, start=1):
        key = _doc_identity(doc)
        fused.setdefault(key, {"doc": doc, "score": 0.0, "channels": set()})
        fused[key]["score"] += 1.0 / (rrf_constant + rank)
        fused[key]["channels"].add("BM25")
    return [item["doc"] for item in sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:k]]


def rerank_documents(llm, question, documents, top_n=5):
    """Use the chat model as a relevance reranker; return original Document objects."""
    if not documents:
        return [], 0.0
    candidates = documents[:min(len(documents), 12)]
    numbered = []
    for idx, doc in enumerate(candidates):
        text_value = re.sub(r"\s+", " ", doc.page_content).strip()[:1800]
        numbered.append(f"[{idx}] {text_value}")
    grading_prompt = (
        "You are a retrieval relevance judge. Score each passage from 0 to 100 for how directly "
        "it helps answer the user question. Reward factual overlap and exact scheme/eligibility details. "
        "Do not reward mere topic similarity. Return ONLY valid JSON as "
        '{"scores":[{"id":0,"score":0}]} with one item per passage.\n'
        f"QUESTION: {question}\nPASSAGES:\n" + "\n\n".join(numbered)
    )
    try:
        result = llm.invoke(grading_prompt)
        raw = result.content if hasattr(result, "content") else str(result)
        match = re.search(r"\{.*\}", str(raw), flags=re.S)
        payload = json.loads(match.group(0)) if match else {}
        score_map = {
            int(item["id"]): max(0.0, min(100.0, float(item["score"])))
            for item in payload.get("scores", [])
            if isinstance(item, dict) and "id" in item and "score" in item
        }
        ranked = sorted(
            enumerate(candidates),
            key=lambda pair: score_map.get(pair[0], 0.0),
            reverse=True,
        )
        ranked_docs = [doc for idx, doc in ranked if idx in score_map]
        # If grading output was incomplete, retain the remaining candidates in retrieval order.
        ranked_docs.extend(doc for idx, doc in enumerate(candidates) if idx not in score_map)
        top_score = max(score_map.values(), default=0.0)
        return ranked_docs[:max(1, int(top_n))], top_score
    except Exception:
        # Retrieval must remain usable if the model returns malformed grading output.
        return candidates[:max(1, int(top_n))], 0.0


def corrective_retrieve(llm, question, vector_db, initial_docs, top_n=5, threshold=35.0, use_hybrid=True):
    """Rerank retrieved passages; if relevance is weak, rewrite and retry retrieval once."""
    ranked_docs, best_score = rerank_documents(llm, question, initial_docs, top_n=max(top_n, 8))
    corrected = False
    if best_score < threshold and vector_db is not None:
        try:
            rewrite_prompt = (
                "Rewrite this question as one concise search query that preserves its intent and "
                "important named entities. Do not answer it. Return only the query.\nQuestion: "
                + question
            )
            rewrite_result = llm.invoke(rewrite_prompt)
            rewritten = str(getattr(rewrite_result, "content", rewrite_result)).strip().strip('"')
            if rewritten and rewritten.lower() != question.strip().lower():
                retry_docs = hybrid_retrieve(rewritten, vector_db, k=30, use_hybrid=use_hybrid)
                existing = {_doc_identity(doc) for doc in initial_docs}
                merged = list(initial_docs) + [doc for doc in retry_docs if _doc_identity(doc) not in existing]
                ranked_docs, best_score = rerank_documents(llm, question, merged, top_n=max(top_n, 8))
                corrected = True
        except Exception:
            pass
    return ranked_docs[:top_n], best_score, corrected

# 2. CUSTOM PROFESSIONAL CSS
# =====================================================

st.markdown(
    """
    <style>
    /* Main page */
    .stApp {
        background: #f3f8ff;
    }

    [data-testid="stHeader"] {
        background: rgba(243, 248, 255, 0.92);
    }

    .block-container {
        max-width: 1080px;
        padding-top: 1.25rem;
        padding-bottom: 4rem;
    }

    /* Arctic Blue SaaS theme */
    [data-testid="stRadio"] [role="radiogroup"] { gap: 8px; }
    [data-testid="stRadio"] label { background:#ffffff; border:1px solid #dbeafe; border-radius:12px; padding:8px 14px; }
    [data-testid="stRadio"] label:has(input:checked) { background:#eff6ff; border-color:#60a5fa; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #eef6ff 100%);
        border-right: 1px solid #dbeafe;
    }

    [data-testid="stSidebar"] * {
        color: #16345c;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #58708e;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(120deg, #0b3b78 0%, #1769aa 62%, #3b82f6 100%);
        border-radius: 16px;
        padding: 22px;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 12px 35px rgba(23, 105, 170, 0.16);
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(26px, 3vw, 34px);
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
        border: 1px solid #dbeafe;
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
        border: 1px solid #dbeafe;
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
        border-color: #60a5fa;
        color: #1769aa;
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
        background: #dbeafe;
        color: #1d4ed8;
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
# 3. SIMPLE WORKSPACE NAVIGATION
# =====================================================
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:14px;margin:0 0 14px 0;">
      <div style="background:#e7f1ff;border:1px solid #cfe1ff;border-radius:14px;padding:12px;font-size:24px;">🌾</div>
      <div>
        <div style="font-size:12px;font-weight:800;letter-spacing:.08em;color:#1769aa;">AGRIDOCS AI</div>
        <div style="font-size:14px;color:#64748b;">Ask questions across your agricultural documents</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

wizard_steps = ["1 | Upload", "2 | Configure", "3 | Retrieval", "4 | Chat"]
if "wizard_step" not in st.session_state or st.session_state.wizard_step not in wizard_steps:
    st.session_state.wizard_step = wizard_steps[0]

# Simple horizontal workspace options instead of a dropdown.
workspace_labels = {
    wizard_steps[0]: "📤 Upload",
    wizard_steps[1]: "⚙️ Configure",
    wizard_steps[2]: "🔎 Retrieve",
    wizard_steps[3]: "💬 Chat",
}
is_processing = bool(st.session_state.get("agri_processing", False))
if is_processing:
    st.warning("⏳ The app is marked as processing. If no progress bar is currently moving, the previous run may have been interrupted.")
    if st.button("Reset stuck processing state", key="reset_stuck_processing"):
        st.session_state["agri_processing"] = False
        st.rerun()

selected_step = st.radio(
    "Workspace",
    options=wizard_steps,
    format_func=lambda step: workspace_labels[step],
    key="wizard_step",
    horizontal=True,
    label_visibility="collapsed",
    disabled=is_processing,
)
step_index = wizard_steps.index(selected_step)
step_descriptions = {
    wizard_steps[0]: ("Add your documents", "Choose a folder of PDFs or add supported files."),
    wizard_steps[1]: ("Configure indexing", "Use the default chunk settings, or adjust them if needed."),
    wizard_steps[2]: ("Search settings", "Optional filters and retrieval tuning."),
    wizard_steps[3]: ("Chat with your sources", "Ask a question and review the cited sources."),
}
step_title, step_help = step_descriptions[selected_step]
st.markdown(
    f"""<div style="background:#ffffff;border:1px solid #dbeafe;border-radius:14px;padding:16px 20px;margin:6px 0 18px 0;">
    <div style="font-size:21px;font-weight:750;color:#102a4c;margin-bottom:4px;">{step_title}</div>
    <div style="font-size:14px;color:#64748b;">{step_help}</div></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;padding:10px 0 18px 0;">
            <div style="width:54px;height:54px;flex:0 0 54px;border-radius:16px;background:linear-gradient(145deg,#e7f8ec,#c7efd2);display:flex;align-items:center;justify-content:center;border:1px solid #b8e3c4;">
                <svg viewBox="0 0 64 64" width="42" height="42" role="img" aria-label="Agriculture logo">
                    <path d="M31 52V29" stroke="#247447" stroke-width="4" stroke-linecap="round" fill="none"/>
                    <path d="M31 34C14 34 10 23 11 12c12 0 24 5 24 18" fill="#48a868" stroke="#247447" stroke-width="2.5" stroke-linejoin="round"/>
                    <path d="M33 27C33 12 44 7 55 8c0 13-6 23-22 24" fill="#80c995" stroke="#247447" stroke-width="2.5" stroke-linejoin="round"/>
                    <path d="M18 48h28M23 54h18" stroke="#247447" stroke-width="3" stroke-linecap="round"/>
                </svg>
            </div>
            <div>
                <div style="font-size:21px;font-weight:800;color:#12345a;line-height:1.2;">AgriDocs AI</div>
                <div style="font-size:10px;font-weight:750;letter-spacing:.09em;color:#27804a;margin-top:5px;">AGRICULTURE KNOWLEDGE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Explicit two-step reset to avoid accidental deletion of the local RAG store.
    st.markdown("### Knowledge Base Maintenance")
    if st.session_state.pop("reset_knowledge_base_completed", False):
        st.success("Knowledge base reset successfully. Upload/process your PDFs and fetch URLs again.")
    if st.button("Reset Knowledge Base", key="reset_knowledge_base_start", width="stretch", type="secondary"):
        st.session_state["reset_knowledge_base_pending"] = True
        st.rerun()

    if st.session_state.get("reset_knowledge_base_pending", False):
        st.warning(
            "This permanently removes the managed FAISS index, the legacy FAISS index, "
            "and document_store data (manifest, extracted text, tables, images, previews, and chunk cache). "
            "It does not delete your original PDF files or your .env file. You will need to upload/process "
            "documents and fetch webpages again."
        )
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            confirm_reset = st.button(
                "Yes, reset everything", key="reset_knowledge_base_confirm", type="primary", width="stretch"
            )
        with cancel_col:
            cancel_reset = st.button(
                "Cancel", key="reset_knowledge_base_cancel", width="stretch"
            )

        if cancel_reset:
            st.session_state["reset_knowledge_base_pending"] = False
            st.rerun()

        if confirm_reset:
            paths_to_remove = [Path("faiss_index_uploaded"), Path("faiss_index"), DOCUMENT_STORE_DIR]
            try:
                for path in paths_to_remove:
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.exists():
                        path.unlink()

                # Clear in-memory references so this process cannot continue querying a stale index.
                for state_key in (
                    "active_vector_db", "active_index_label", "messages",
                    "pdf_visual_pages", "pdf_embedded_images", "pdf_extracted_tables",
                    "chunk_preview", "chunk_total", "reset_knowledge_base_pending",
                    "pdf_uploader", "other_document_uploader",
                ):
                    st.session_state.pop(state_key, None)

                st.session_state["messages"] = []
                st.session_state["reset_knowledge_base_completed"] = True
                st.rerun()
            except Exception as exc:
                st.error(f"Reset failed while removing local index data: {exc}")
                st.caption("Close other processes that may be locking these files, then retry the reset.")

    st.divider()
    st.markdown("### Knowledge Sources")
    if selected_step == wizard_steps[0]:
        st.markdown("#### 📁 Load PDFs from a folder")
        pdf_directory_text = st.text_input(
            "PDF directory path",
            value=st.session_state.get("pdf_directory_path", ""),
            placeholder=r"E:\AI\documents",
            key="pdf_directory_path",
            help="Enter the local folder path. All PDF files directly inside this folder will be discovered automatically.",
        ).strip().strip('"')
        uploaded_pdfs = []
        pdf_directory = Path(pdf_directory_text).expanduser() if pdf_directory_text else None
        if pdf_directory_text:
            if pdf_directory and pdf_directory.exists() and pdf_directory.is_dir():
                pdf_paths = sorted(
                    (p for p in pdf_directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
                    key=lambda p: p.name.lower(),
                )
                uploaded_pdfs = [LocalPDFFile(p) for p in pdf_paths]
                if uploaded_pdfs:
                    st.success(f"Found {len(uploaded_pdfs)} PDF file(s) in `{pdf_directory}`.")
                    with st.expander("View discovered PDFs"):
                        for pdf_file in uploaded_pdfs:
                            try:
                                size_mb = pdf_file.path.stat().st_size / (1024 * 1024)
                            except OSError:
                                size_mb = 0
                            st.write(f"📄 {pdf_file.name} — {size_mb:.2f} MB")
                else:
                    st.warning("No PDF files were found directly inside this directory.")
            else:
                st.error("Directory not found or not accessible. Check the path and try again.")
        st.caption("The app scans the selected folder (not subfolders) and processes every PDF found there.")
        uploaded_other_files = st.file_uploader(
            "Upload DOCX, TXT, Markdown, CSV, or Excel files",
            type=["docx", "txt", "md", "markdown", "csv", "xlsx", "xls"],
            accept_multiple_files=True, key="other_document_uploader",
            help="These formats are normalized into the same RAG knowledge base.",
        )
        process_other_files = st.button(
            "Process Other Documents", width="stretch",
            disabled=not bool(uploaded_other_files) or is_processing,
            key="process_other_documents",
        )
        describe_images = st.checkbox(
            "Generate AI descriptions for images and charts (uses API tokens)",
            value=False, key="describe_images_setting",
            help="Descriptions are saved with image metadata and added to searchable PDF text.",
        )
        process_pdfs = st.button(
            "Process PDFs in Directory", width="stretch", type="primary",
            disabled=not bool(uploaded_pdfs) or is_processing,
            key="process_uploaded_pdfs",
        )
        st.caption("Limits: 50 MB/file | 2,000 pages/PDF | OCR on up to 30 pages/PDF")
        st.caption("Incremental indexing reuses extraction metadata for unchanged sources.")
        st.caption("PDF hyperlinks are collected during processing and can be fetched below.")
    else:
        # File upload widgets are not rendered outside Step 1; initialize safe defaults for this run.
        uploaded_pdfs = []
        uploaded_other_files = []
        process_other_files = False
        process_pdfs = False
        describe_images = bool(st.session_state.get("describe_images_setting", False))
        st.caption("Choose **📤 Upload** above to select a PDF directory or add other documents.")

    if selected_step == wizard_steps[1]:
        st.divider()
        st.markdown("### Text Splitter Settings")
        chunk_size = st.slider("Chunk size (characters)", min_value=250, max_value=2500, value=int(st.session_state.get("saved_chunk_size", 1000)), step=50, key="chunk_size_setting")
        chunk_overlap = st.slider("Chunk overlap (characters)", min_value=0, max_value=min(500, chunk_size - 1), value=min(int(st.session_state.get("saved_chunk_overlap", 150)), chunk_size - 1), step=25, key="chunk_overlap_setting")
        splitter_strategy = st.selectbox("Splitting strategy", ["Recursive", "Character", "Token", "Semantic"], index=["Recursive", "Character", "Token", "Semantic"].index(st.session_state.get("saved_splitter_strategy", "Recursive")), help="Semantic mode compares adjacent sentence embeddings and may take longer.", key="splitter_strategy_setting")
        parent_child = st.checkbox("Use parent-child chunks", value=bool(st.session_state.get("saved_parent_child", False)), key="parent_child_setting", help="Index smaller child chunks and retain larger parent context.")
        if splitter_strategy == "Token":
            token_chunk_size = st.slider("Token chunk size", 100, 800, int(st.session_state.get("saved_token_chunk_size", 350)), 25, key="token_chunk_size_setting")
            token_overlap = st.slider("Token overlap", 0, 120, int(st.session_state.get("saved_token_overlap", 40)), 5, key="token_overlap_setting")
        else:
            token_chunk_size, token_overlap = 350, 40
        st.session_state.saved_chunk_size = chunk_size
        st.session_state.saved_chunk_overlap = chunk_overlap
        st.session_state.saved_splitter_strategy = splitter_strategy
        st.session_state.saved_parent_child = parent_child
        st.session_state.saved_token_chunk_size = token_chunk_size
        st.session_state.saved_token_overlap = token_overlap
        st.caption("Heading metadata and table header-aware row grouping are applied automatically.")
    else:
        chunk_size = int(st.session_state.get("saved_chunk_size", 1000))
        chunk_overlap = min(int(st.session_state.get("saved_chunk_overlap", 150)), chunk_size - 1)
        splitter_strategy = st.session_state.get("saved_splitter_strategy", "Recursive")
        parent_child = bool(st.session_state.get("saved_parent_child", False))
        token_chunk_size = int(st.session_state.get("saved_token_chunk_size", 350)) if splitter_strategy == "Token" else 350
        token_overlap = int(st.session_state.get("saved_token_overlap", 40)) if splitter_strategy == "Token" else 40
    chunk_settings = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "splitter_strategy": splitter_strategy, "parent_child": parent_child, "parent_chunk_size": max(chunk_size * 2, 1000), "token_chunk_size": token_chunk_size, "token_overlap": token_overlap}

    if selected_step == wizard_steps[2]:
        st.divider()
        st.markdown("### Retrieval Filters")
        filter_manifest = read_manifest()
        filter_pdf_names = sorted(filter_manifest.get("documents", {}).keys())
        filter_urls = sorted(filter_manifest.get("webpages", {}).keys())
        selected_pdf_filter = st.multiselect("PDF filter (empty = all)", filter_pdf_names, default=st.session_state.get("saved_pdf_filter", []), key="retrieval_pdf_filter")
        document_type_filter = st.selectbox("Document type", ["All", "PDF pages", "Tables", "Webpages", "Other documents"], index=["All", "PDF pages", "Tables", "Webpages", "Other documents"].index(st.session_state.get("saved_document_type_filter", "All")), key="retrieval_type_filter")
        max_known_page = max([len(r.get("pages", [])) for r in filter_manifest.get("documents", {}).values()] + [1])
        page_start = st.number_input("Page number from", min_value=1, max_value=max(1, max_known_page), value=min(int(st.session_state.get("saved_page_start", 1)), max(1, max_known_page)), step=1, key="retrieval_page_start")
        page_end = st.number_input("Page number to (0 = no upper limit)", min_value=0, max_value=max_known_page, value=min(int(st.session_state.get("saved_page_end", 0)), max_known_page), step=1, key="retrieval_page_end")
        selected_url_filter = st.multiselect("Webpage URL filter (empty = all)", filter_urls, default=st.session_state.get("saved_url_filter", []), key="retrieval_url_filter")
        st.session_state.saved_pdf_filter = selected_pdf_filter
        st.session_state.saved_document_type_filter = document_type_filter
        st.session_state.saved_page_start = page_start
        st.session_state.saved_page_end = page_end
        st.session_state.saved_url_filter = selected_url_filter
        with st.expander("Advanced search options", expanded=False):
            use_hybrid_search = st.checkbox("Hybrid search (BM25 + FAISS)", value=st.session_state.get("saved_hybrid_search", True), key="hybrid_search_setting", help="Combines keyword and semantic search.")
            use_reranking = st.checkbox("Rerank results for relevance", value=st.session_state.get("saved_reranking", True), key="reranking_setting", help="Uses the answer model to rank candidate passages.")
            use_corrective_rag = st.checkbox("Retry weak searches (Corrective RAG)", value=st.session_state.get("saved_corrective_rag", True), key="corrective_rag_setting", help="Rewrites the search query and retries once if matches look weak.")
            rerank_top_n = st.slider("Passages used for each answer", min_value=2, max_value=10, value=int(st.session_state.get("saved_rerank_top_n", 5)), key="rerank_top_n_setting")
            corrective_threshold = st.slider("Retry threshold", min_value=10, max_value=80, value=int(st.session_state.get("saved_corrective_threshold", 35)), step=5, key="corrective_threshold_setting")
        st.session_state.saved_hybrid_search = use_hybrid_search
        st.session_state.saved_reranking = use_reranking
        st.session_state.saved_corrective_rag = use_corrective_rag
        st.session_state.saved_rerank_top_n = rerank_top_n
        st.session_state.saved_corrective_threshold = corrective_threshold
    else:
        selected_pdf_filter = st.session_state.get("saved_pdf_filter", [])
        document_type_filter = st.session_state.get("saved_document_type_filter", "All")
        page_start = int(st.session_state.get("saved_page_start", 1))
        page_end = int(st.session_state.get("saved_page_end", 0))
        selected_url_filter = st.session_state.get("saved_url_filter", [])
        use_hybrid_search = st.session_state.get("saved_hybrid_search", True)
        use_reranking = st.session_state.get("saved_reranking", True)
        use_corrective_rag = st.session_state.get("saved_corrective_rag", True)
        rerank_top_n = int(st.session_state.get("saved_rerank_top_n", 5))
        corrective_threshold = float(st.session_state.get("saved_corrective_threshold", 35))

    st.divider()
    if st.button("Clear Conversation", width="stretch", key="clear_conversation_button"):
        st.session_state.messages = []
        st.rerun()
    st.markdown(
        """
        <div style="margin-top:24px;padding:13px;border:1px solid #dbeafe;border-radius:12px;background:#f8fbff;">
          <div style="font-weight:800;margin-bottom:5px;color:#12345a;">Private workspace</div>
          <div style="font-size:12px;line-height:1.5;color:#64748b;">Your API key is read from your local environment. Keep secrets out of source control.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# 4. COMPACT PAGE HEADER
# =====================================================
st.markdown(
    """
    <div class="hero" style="padding:22px 26px;border-radius:16px;margin-bottom:18px;background:linear-gradient(120deg,#14532d 0%,#23834d 58%,#65a765 100%);">
        <div style="display:flex;align-items:center;gap:14px;">
            <div style="width:58px;height:58px;flex:0 0 58px;border-radius:17px;background:rgba(255,255,255,.96);display:flex;align-items:center;justify-content:center;">
                <svg viewBox="0 0 64 64" width="46" height="46" role="img" aria-label="AgriDocs agriculture logo">
                    <path d="M31 53V29" stroke="#247447" stroke-width="4" stroke-linecap="round" fill="none"/>
                    <path d="M31 34C14 34 10 23 11 12c12 0 24 5 24 18" fill="#48a868" stroke="#247447" stroke-width="2.5" stroke-linejoin="round"/>
                    <path d="M33 27C33 12 44 7 55 8c0 13-6 23-22 24" fill="#80c995" stroke="#247447" stroke-width="2.5" stroke-linejoin="round"/>
                    <path d="M18 48h28M23 54h18" stroke="#247447" stroke-width="3" stroke-linecap="round"/>
                </svg>
            </div>
            <div>
                <span class="hero-label">AGRICULTURE KNOWLEDGE ASSISTANT</span>
                <h1 style="font-size:30px;margin:10px 0 6px 0;">AgriDocs AI Assistant</h1>
                <p style="font-size:14px;">Ask about farming schemes, guidance and documents — with source references.</p>
            </div>
        </div>
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
# 6. LOAD EMBEDDINGS, OPTIONAL EXISTING FAISS DATABASE, AND LLM
# =====================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # Larger batches can speed up embedding many chunks on capable machines.
        encode_kwargs={"batch_size": 32},
    )


@st.cache_resource
def load_vector_database(index_folder="faiss_index"):
    """Load a trusted local FAISS index from the given folder."""
    return FAISS.load_local(
        index_folder,
        load_embeddings(),
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Add OPENAI_API_KEY to your .env file.")
    st.stop()

try:
    llm = load_llm()
except Exception as e:
    st.error("Unable to initialize the language model. Check your API key and connection.")
    st.exception(e)
    st.stop()

# Load the persistent manifest and restore visual/table metadata after app restarts.
manifest = read_manifest()
visual_pages, embedded_images, extracted_tables = manifest_assets(manifest)
st.session_state.pdf_visual_pages = visual_pages
st.session_state.pdf_embedded_images = embedded_images
st.session_state.pdf_extracted_tables = extracted_tables

# Prefer the index built from the persistent document store when it exists.
vector_db = st.session_state.get("active_vector_db")
active_index_label = st.session_state.get("active_index_label", "")
if vector_db is None:
    if Path("faiss_index_uploaded/index.faiss").exists() and manifest.get("documents"):
        try:
            with st.spinner("Loading your persistent uploaded-PDF knowledge base..."):
                vector_db = load_vector_database("faiss_index_uploaded")
                active_index_label = "Uploaded PDF index"
        except Exception as e:
            st.warning("The uploaded PDF index could not be loaded. Reprocess your PDFs to rebuild it.")
            st.caption(str(e))
            vector_db = None
    elif Path("faiss_index/index.faiss").exists():
        try:
            with st.spinner("Connecting to your existing PDF knowledge base..."):
                vector_db = load_vector_database("faiss_index")
                active_index_label = "Existing FAISS index"
        except Exception as e:
            st.warning("The existing FAISS index could not be loaded. You can upload and process PDFs below.")
            st.caption(str(e))
            vector_db = None
    elif Path("faiss_index_uploaded/index.faiss").exists():
        try:
            with st.spinner("Loading the previously uploaded PDF index..."):
                vector_db = load_vector_database("faiss_index_uploaded")
                active_index_label = "Uploaded PDF index"
        except Exception as e:
            st.warning("The previously uploaded PDF index could not be loaded.")
            st.caption(str(e))
            vector_db = None

# Process uploads only after the user clicks the sidebar button.
if process_pdfs and uploaded_pdfs:
    DOCUMENT_STORE_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_STORE_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest()
    manifest.setdefault("version", 1)
    manifest.setdefault("documents", {})
    report_rows = []
    uploaded_hashes = set()

    st.session_state["agri_processing"] = True
    pdf_progress = st.progress(0, text="Preparing PDF extraction...")
    with st.status("Validating and extracting PDFs from the selected directory...", expanded=True) as status:
        total_pdf_files = max(1, len(uploaded_pdfs))
        for pdf_file_index, uploaded_file in enumerate(uploaded_pdfs):
            pdf_progress.progress(pdf_file_index / total_pdf_files, text=f"Preparing {pdf_file_index + 1}/{total_pdf_files}: {uploaded_file.name}")
            filename = safe_source_name(uploaded_file.name)
            pdf_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(pdf_bytes).hexdigest()

            if not pdf_bytes:
                report_rows.append({"File": filename, "Status": "Rejected: empty (0 bytes)", "Pages": 0, "OCR pages": 0, "Images": 0, "Tables": 0, "Text characters": 0})
                continue
            if len(pdf_bytes) > MAX_FILE_MB * 1024 * 1024:
                report_rows.append({"File": filename, "Status": f"Rejected: over {MAX_FILE_MB} MB", "Pages": 0, "OCR pages": 0, "Images": 0, "Tables": 0, "Text characters": 0})
                continue
            if file_hash in uploaded_hashes:
                report_rows.append({"File": filename, "Status": "Skipped: duplicate content in this upload batch", "Pages": 0, "OCR pages": 0, "Images": 0, "Tables": 0, "Text characters": 0})
                continue
            uploaded_hashes.add(file_hash)

            existing = manifest["documents"].get(filename)
            if existing and existing.get("sha256") == file_hash and "urls" in existing and (not describe_images or existing.get("processed_with_vision")):
                report_rows.append({
                    "File": filename,
                    "Status": "Skipped extraction: unchanged file reused from cache",
                    "Pages": len(existing.get("pages", [])),
                    "OCR pages": existing.get("ocr_pages", 0),
                    "Images": len(existing.get("images", [])),
                    "Tables": len(existing.get("tables", [])),
                    "Text characters": existing.get("text_characters", 0),
                    "URLs found": len(existing.get("urls", [])),
                })
                continue

            # If this filename has changed, replace its manifest entry. Old asset files are
            # left on disk intentionally; they are harmless and can be cleaned up separately.
            try:
                pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            except Exception as exc:
                report_rows.append({"File": filename, "Status": f"Rejected: invalid PDF ({type(exc).__name__})", "Pages": 0, "OCR pages": 0, "Images": 0, "Tables": 0, "Text characters": 0})
                continue

            doc_record = {
                "sha256": file_hash,
                "file_size_bytes": len(pdf_bytes),
                "pages": [],
                "images": [],
                "visual_pages": [],
                "tables": [],
                "urls": [],
                "ocr_pages": 0,
                "text_characters": 0,
                "processed_with_vision": bool(describe_images),
            }
            page_errors = 0
            vision_count = 0

            with pdf:
                page_count = len(pdf)
                toc_entries = pdf.get_toc() or []
                toc_by_page = []
                for toc_entry in toc_entries:
                    if len(toc_entry) >= 3:
                        toc_by_page.append((int(toc_entry[2]), int(toc_entry[0]), str(toc_entry[1]).strip()))
                if pdf.needs_pass:
                    report_rows.append({"File": filename, "Status": "Rejected: password-protected PDF", "Pages": page_count, "OCR pages": 0, "Images": 0, "Tables": 0, "Text characters": 0})
                    continue
                if page_count == 0:
                    report_rows.append({"File": filename, "Status": "Rejected: no pages", "Pages": 0, "OCR pages": 0, "Images": 0, "Tables": 0, "Text characters": 0})
                    continue
                if page_count > MAX_PAGES_PER_PDF:
                    report_rows.append({"File": filename, "Status": f"Rejected: over {MAX_PAGES_PER_PDF} pages", "Pages": page_count, "OCR pages": 0, "Images": 0, "Tables": 0, "Text characters": 0})
                    continue

                for page_index in range(page_count):
                    overall_page_progress = (pdf_file_index + (page_index / max(1, page_count))) / total_pdf_files
                    pdf_progress.progress(min(0.95, overall_page_progress), text=f"Extracting {filename} — page {page_index + 1}/{page_count}")
                    page = pdf[page_index]
                    page_number = page_index + 1
                    page_text = (page.get_text("text", sort=True) or "").strip()
                    for found_url in extract_urls_from_page(page, page_text):
                        if found_url not in doc_record["urls"] and len(doc_record["urls"]) < MAX_URLS_PER_PDF:
                            doc_record["urls"].append(found_url)
                    heading = detect_heading(page)
                    # Prefer the PDF's built-in outline when present, preserving nested sections.
                    active_outline = []
                    for toc_entry in toc_entries:
                        if len(toc_entry) >= 3 and int(toc_entry[2]) <= page_number:
                            level = max(1, int(toc_entry[0]))
                            title = str(toc_entry[1]).strip()
                            active_outline = [entry for entry in active_outline if entry[0] < level]
                            active_outline.append((level, title))
                    section_path = " > ".join(title for _, title in active_outline) or heading
                    extraction_method = "native text"
                    page_tables = []
                    page_images = []
                    table_text_parts = []
                    page_image_path = ""

                    # 1) Table-aware extraction: store rows structurally and make them searchable.
                    try:
                        table_finder = page.find_tables()
                        for table_number, table in enumerate(table_finder.tables, start=1):
                            rows = table.extract() or []
                            cleaned_rows = [
                                [str(cell).strip() if cell is not None else "" for cell in row]
                                for row in rows
                            ]
                            if cleaned_rows and any(cell for row in cleaned_rows for cell in row):
                                table_record = {
                                    "source": filename,
                                    "page_number": page_number,
                                    "table_number": table_number,
                                    "rows": cleaned_rows,
                                }
                                page_tables.append(table_record)
                                doc_record["tables"].append(table_record)
                                table_text_parts.append(
                                    f"Table {table_number} on page {page_number}:\n" +
                                    "\n".join(" | ".join(row) for row in cleaned_rows)
                                )
                    except Exception:
                        # Continue even if the installed PyMuPDF version cannot detect tables.
                        pass
                    # Extracted tables are stored separately and indexed as header-aware table documents.
                    # This avoids flattening table cells into the surrounding paragraph text.
                    text_length_before_vision = len(page_text)

                    image_entries = page.get_images(full=True)
                    has_embedded_images = bool(image_entries)
                    low_text_page = len(page_text) < MIN_TEXT_CHARS_BEFORE_OCR
                    try:
                        # Vector charts/diagrams may have no embedded raster image objects.
                        has_vector_graphics = len(page.get_drawings()) >= 20
                    except Exception:
                        has_vector_graphics = False
                    unique_xrefs = set()
                    for page_image_number, image_entry in enumerate(image_entries[:MAX_EMBEDDED_IMAGES_PER_PAGE], start=1):
                        xref = image_entry[0]
                        if not xref or xref in unique_xrefs:
                            continue
                        unique_xrefs.add(xref)
                        try:
                            image_info = pdf.extract_image(xref)
                            raw_image = image_info.get("image")
                            if not raw_image:
                                continue
                            image_path = EMBEDDED_IMAGES_DIR / f"{file_hash[:16]}_p{page_number}_i{page_image_number}.png"
                            with Image.open(io.BytesIO(raw_image)) as extracted_image:
                                extracted_image.load()
                                if extracted_image.mode not in ("RGB", "RGBA"):
                                    extracted_image = extracted_image.convert("RGB")
                                extracted_image.save(image_path, format="PNG")
                            image_record = {
                                "source": filename,
                                "page_number": page_number,
                                "page_image_number": page_image_number,
                                "image_path": str(image_path),
                                "description": "",
                            }
                            page_images.append(image_record)
                            doc_record["images"].append(image_record)
                        except Exception:
                            continue

                    # 2) Page preview fallback for scanned pages/charts/visual context.
                    if has_embedded_images or low_text_page or has_vector_graphics:
                        try:
                            rect = page.rect
                            scale = min(120 / 72, MAX_PREVIEW_DIMENSION / max(rect.width, rect.height))
                            estimated_pixels = (rect.width * scale) * (rect.height * scale)
                            if estimated_pixels > MAX_PREVIEW_PIXELS:
                                scale *= (MAX_PREVIEW_PIXELS / estimated_pixels) ** 0.5
                            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                            preview_path = VISUALS_DIR / f"{file_hash[:16]}_page_{page_number}.png"
                            pix.save(str(preview_path))
                            page_image_path = str(preview_path)
                            doc_record["visual_pages"].append({
                                "source": filename,
                                "page_number": page_number,
                                "image_path": page_image_path,
                            })
                        except Exception:
                            page_image_path = ""
                            page_errors += 1

                    # 3) Optional vision descriptions are saved as searchable text.
                    # Opt-in avoids surprise API usage; only the first N images per PDF are described.
                    if describe_images:
                        for image_record in page_images:
                            if vision_count >= MAX_VISION_IMAGES_PER_PDF:
                                break
                            try:
                                description = describe_image_with_vision(
                                    llm, image_record["image_path"], filename, page_number
                                )
                                image_record["description"] = description
                                if description:
                                    page_text += f"\n\nImage description (image on page {page_number}): {description}"
                                vision_count += 1
                            except Exception as exc:
                                page_errors += 1
                                image_record["description_error"] = type(exc).__name__
                        # If the page has no extractable image object, describe its preview instead.
                        if not page_images and page_image_path and vision_count < MAX_VISION_IMAGES_PER_PDF:
                            try:
                                description = describe_image_with_vision(llm, page_image_path, filename, page_number)
                                if description:
                                    page_text += f"\n\nPage visual description (page {page_number}): {description}"
                                vision_count += 1
                            except Exception:
                                page_errors += 1

                    # 4) OCR fallback. Keep extracted native text and use OCR only if it adds more.
                    if low_text_page and doc_record["ocr_pages"] < MAX_OCR_PAGES_PER_PDF:
                        try:
                            rect = page.rect
                            scale = 200 / 72
                            if max(rect.width, rect.height) * scale > 5000:
                                scale = 5000 / max(rect.width, rect.height)
                            estimated_pixels = (rect.width * scale) * (rect.height * scale)
                            if estimated_pixels > 16_000_000:
                                scale *= (16_000_000 / estimated_pixels) ** 0.5
                            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                            ocr_image = Image.open(io.BytesIO(pix.tobytes("png")))
                            ocr_text = pytesseract.image_to_string(
                                ocr_image, lang="eng", timeout=OCR_TIMEOUT_SECONDS
                            ).strip()
                            doc_record["ocr_pages"] += 1
                            if ocr_text and len(ocr_text) > text_length_before_vision and ocr_text not in page_text:
                                had_native_text = text_length_before_vision > 0
                                page_text = (page_text + "\n\nOCR text:\n" + ocr_text).strip() if page_text else ocr_text
                                extraction_method = "native text + OCR" if had_native_text else "OCR"
                        except Exception:
                            page_errors += 1

                    if heading:
                        # Keep the heading visible in text context and in metadata for citations.
                        if heading.lower() not in page_text[:300].lower():
                            page_text = f"Section/page heading: {heading}\n\n{page_text}".strip()
                    doc_record["text_characters"] += len(page_text)
                    doc_record["pages"].append({
                        "page_number": page_number,
                        "page_text": page_text,
                        "heading": heading,
                        "section_path": section_path,
                        "extraction_method": extraction_method,
                        "page_image": page_image_path,
                    })

            # Backfill a section path when a PDF has no built-in outline.
            infer_heading_hierarchy(doc_record.get("pages", []))
            # Persist text and table sidecars as well as the central manifest.
            (TEXT_STORE_DIR / f"{file_hash[:16]}.json").write_text(
                json.dumps(doc_record["pages"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (TABLES_DIR / f"{file_hash[:16]}.json").write_text(
                json.dumps(doc_record["tables"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
            manifest["documents"][filename] = doc_record
            report_rows.append({
                "File": filename,
                "Status": "Extracted with page error(s)" if page_errors else "Extracted successfully",
                "Pages": page_count,
                "OCR pages": doc_record["ocr_pages"],
                "Images": len(doc_record["images"]),
                "Tables": len(doc_record["tables"]),
                "Text characters": doc_record["text_characters"],
                "URLs found": len(doc_record.get("urls", [])),
            })

        # Save metadata first. It allows image/table browsing even when no text chunks exist.
        write_manifest(manifest)
        visual_pages, embedded_images, extracted_tables = manifest_assets(manifest)
        st.session_state.pdf_visual_pages = visual_pages
        st.session_state.pdf_embedded_images = embedded_images
        st.session_state.pdf_extracted_tables = extracted_tables

        # Rebuild the managed index from all cached PDF pages and previously fetched webpages.
        # The original user's faiss_index directory remains untouched.
        status.update(label="Chunking documents and rebuilding the searchable index...", state="running")
        try:
            new_vector_db, chunk_count, generated_chunks = rebuild_uploaded_index(manifest, chunk_settings)
            if new_vector_db is not None:
                st.session_state.active_vector_db = new_vector_db
                st.session_state.active_index_label = "Uploaded PDF index"
                vector_db = new_vector_db
                active_index_label = "Uploaded PDF index"
                st.session_state.chunk_preview = generated_chunks[:100]
                st.session_state.chunk_total = chunk_count
                st.session_state.messages = []
                status.update(label=f"Ready! Indexed {chunk_count} chunks from {len(manifest['documents'])} cached PDF(s) and {len(manifest.get('webpages', {}))} webpage(s).", state="complete")
            else:
                vector_db = None
                st.session_state.active_vector_db = None
                status.update(label="No searchable text was found. Images and tables may still be browsable below.", state="complete")
        except Exception as exc:
            status.update(label="Index creation failed.", state="error")
            st.error("Could not create the FAISS index. Check the error below.")
            st.exception(exc)

    pdf_progress.progress(1.0, text="PDF extraction and indexing finished.")
    st.session_state["agri_processing"] = False
    if report_rows:
        st.markdown("### Report PDF processing report")
        st.dataframe(report_rows, width="stretch", hide_index=True)
    if vector_db is not None and active_index_label == "Uploaded PDF index":
        st.success("Your PDF assets and manifest are saved in `document_store/`. The original `faiss_index` folder was not changed.")

# Process additional document formats through the same manifest and index.
if process_other_files and uploaded_other_files:
    st.session_state["agri_processing"] = True
    other_progress = st.progress(0, text="Preparing document processing...")
    manifest = read_manifest()
    manifest.setdefault("other_documents", {})
    report = []
    seen_hashes = set()
    total_other_files = max(1, len(uploaded_other_files))
    for other_file_index, uploaded_file in enumerate(uploaded_other_files):
        other_progress.progress(other_file_index / total_other_files, text=f"Loading document {other_file_index + 1}/{total_other_files}: {uploaded_file.name}")
        filename = safe_source_name(uploaded_file.name)
        raw = uploaded_file.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        if not raw or len(raw) > MAX_FILE_MB * 1024 * 1024:
            report.append({"File": filename, "Status": "Rejected: empty or exceeds 50 MB"})
            continue
        if digest in seen_hashes:
            report.append({"File": filename, "Status": "Skipped: duplicate upload content"})
            continue
        seen_hashes.add(digest)
        existing = manifest["other_documents"].get(filename)
        if existing and existing.get("sha256") == digest:
            report.append({"File": filename, "Status": "Skipped: unchanged file reused from cache"})
            continue
        try:
            record = load_non_pdf_file(filename, raw)
            manifest["other_documents"][filename] = record
            report.append({"File": filename, "Status": "Loaded", "Type": record["source_type"], "Documents": len(record["documents"]), "Text characters": record["text_characters"]})
        except Exception as exc:
            report.append({"File": filename, "Status": f"Failed: {type(exc).__name__}: {exc}"})
    write_manifest(manifest)
    other_progress.progress(0.9, text="Rebuilding searchable index...")
    st.markdown("### Report Document processing report")
    st.dataframe(report, width="stretch", hide_index=True)
    try:
        with st.spinner("Splitting documents and rebuilding the managed FAISS index..."):
            new_db, count, chunks = rebuild_uploaded_index(manifest, chunk_settings)
        if new_db is not None:
            st.session_state.active_vector_db = new_db
            st.session_state.active_index_label = "Uploaded document index"
            st.session_state.chunk_preview = chunks[:100]
            st.session_state.chunk_total = count
            st.success(f"Indexed {count} chunks across PDFs, webpages, and other documents.")
        else:
            st.warning("No searchable text was found in the selected documents.")
    except Exception as exc:
        st.error("The documents were saved, but index rebuilding failed.")
        st.exception(exc)
    other_progress.progress(1.0, text="Document processing finished.")
    st.session_state["agri_processing"] = False
    st.rerun()

# =====================================================
# 7A. URL DISCOVERY AND WEBPAGE INGESTION
# =====================================================
manifest = read_manifest()
url_to_pdf_sources = {}
for pdf_name, pdf_record in manifest.get("documents", {}).items():
    for url in pdf_record.get("urls", []):
        url_to_pdf_sources.setdefault(url, []).append(pdf_name)

# Keep URL ingestion visible even when the PDF has no embedded hyperlinks.
with st.expander(
    f"🌐 Add webpage URLs (optional) ({len(url_to_pdf_sources)} PDF URL(s) found)",
    expanded=not bool(manifest.get("webpages")),
):
    st.caption(
        "Enter public HTTP/HTTPS webpage URLs below, one per line, or choose links discovered in your PDFs. "
        "Only readable public HTML pages are indexed; redirects, private/local hosts, and non-HTML files are blocked."
    )

    manual_urls_text = st.text_area(
        "Input URL(s)",
        placeholder="https://example.com/page1\nhttps://example.com/page2",
        key="manual_webpage_urls_input",
        help="Paste one URL per line. Comma- or semicolon-separated URLs are also accepted.",
        height=100,
    )

    discovered_urls = sorted(url_to_pdf_sources.keys())
    selected_discovered_urls = []
    if discovered_urls:
        selected_discovered_urls = st.multiselect(
            "Or select URLs discovered in uploaded PDFs",
            options=discovered_urls,
            default=[],
            format_func=lambda u: f"{u}  -  from {', '.join(url_to_pdf_sources.get(u, []))}",
            key="selected_pdf_web_urls",
        )
    else:
        st.info("No embedded URLs were found in the processed PDFs. You can still input webpage URLs above.")

    manual_urls = [u.strip().strip("<>").strip() for u in re.split(r"[\s,;]+", manual_urls_text or "") if u.strip()]
    crawl_internal_links = st.checkbox(
        "Also crawl links inside each website",
        value=True,
        key="crawl_internal_links",
        help="Follows links on the same hostname as each starting URL. External domains are not crawled automatically.",
    )
    crawl_depth = st.select_slider(
        "Link crawl depth (0 = starting pages only)",
        options=[0, 1, 2, 3],
        value=1,
        key="web_crawl_depth",
        help="Depth 1 fetches pages directly linked from each starting URL; depth 2 follows links one level further.",
    )
    max_crawl_pages = st.slider(
        "Maximum total pages to fetch", min_value=1, max_value=50, value=15, step=1,
        key="max_crawl_pages",
        help="Safety limit to avoid crawling a very large website.",
    )
    urls_to_fetch = list(dict.fromkeys(selected_discovered_urls + manual_urls))
    if st.button(
        "Fetch URLs and Add to RAG",
        type="primary",
        disabled=not urls_to_fetch or is_processing,
        key="fetch_pdf_urls",
    ):
        st.session_state["agri_processing"] = True
        crawl_progress = st.progress(0, text="Starting webpage crawl...")
        manifest.setdefault("webpages", {})
        success_count = 0
        failures = []
        # Breadth-first crawl. Every URL is validated by read_webpage_content;
        # linked pages are limited to the seed hostname to avoid domain sprawl.
        queue = [(url, 0, urlparse(url).hostname.lower() if urlparse(url).hostname else "") for url in urls_to_fetch]
        queued = {item[0] for item in queue}
        processed = set()
        page_results = []
        while queue and len(processed) < max_crawl_pages:
            url, depth, seed_host = queue.pop(0)
            crawl_progress.progress(min(0.85, len(processed) / max(1, max_crawl_pages)), text=f"Fetching webpage {len(processed) + 1} (limit {max_crawl_pages})")
            normalized_url = urldefrag(url)[0]
            if normalized_url in processed:
                continue
            processed.add(normalized_url)
            result = read_webpage_content(normalized_url)
            page_results.append((normalized_url, result))
            if result.get("content"):
                manifest["webpages"][normalized_url] = {
                    "title": result.get("title", normalized_url),
                    "url": result.get("url", normalized_url),
                    "content": result["content"],
                    "fetched_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                    "pdf_sources": url_to_pdf_sources.get(normalized_url, []),
                }
                success_count += 1
            else:
                failures.append(f"{normalized_url}: {result.get('error', 'Unknown error')}")
            if crawl_internal_links and depth < crawl_depth:
                for linked_url in result.get("links", []):
                    linked_host = urlparse(linked_url).hostname
                    if not linked_host or linked_host.lower() != seed_host:
                        continue
                    linked_url = urldefrag(linked_url)[0]
                    if linked_url not in queued and linked_url not in processed and len(queued) < max_crawl_pages * 10:
                        queued.add(linked_url)
                        queue.append((linked_url, depth + 1, seed_host))
        if queue:
            failures.append(f"Crawl limit reached: fetched at most {max_crawl_pages} pages. Increase the limit if needed.")
        write_manifest(manifest)
        st.info(f"Crawl finished: visited {len(processed)} page(s), extracted readable text from {success_count} page(s), and found {len(failures)} issue(s).")
        try:
            new_vector_db, chunk_count, generated_chunks = rebuild_uploaded_index(manifest, chunk_settings)
            if new_vector_db is not None:
                st.session_state.active_vector_db = new_vector_db
                st.session_state.active_index_label = "Uploaded document and webpage index"
                st.session_state.chunk_preview = generated_chunks[:100]
                st.session_state.chunk_total = chunk_count
                st.session_state.messages = []
                if success_count:
                    st.success(f"Fetched {success_count} webpage(s). The combined knowledge base now contains {chunk_count} chunks.")
                else:
                    st.warning("No new readable webpage content was fetched. See the errors below.")
            else:
                st.warning("No searchable content was available to index.")
        except Exception as exc:
            st.error("Webpage content was saved, but the index could not be rebuilt.")
            st.exception(exc)
        for failure in failures:
            st.warning(failure)
        crawl_progress.progress(1.0, text="Webpage crawl and indexing finished.")
        st.session_state["agri_processing"] = False
        st.rerun()

    saved_webpages = manifest.get("webpages", {})
    if saved_webpages:
        st.markdown("**Previously fetched webpages**")
        for saved_url, saved_record in saved_webpages.items():
            st.markdown(f"- [{saved_record.get('title', saved_url)}]({saved_url}) - {len(saved_record.get('content', '')):,} characters indexed")

# =====================================================
# 7B. CHUNK PREVIEW, RE-INDEXING, DOCUMENT REMOVAL, EVALUATION
# =====================================================
manifest = read_manifest()

with st.expander("Index tools (optional)", expanded=False):
    st.markdown("#### Current Text Splitter Configuration")
    summary_a, summary_b, summary_c = st.columns(3)
    summary_a.metric("Chunk size", f"{chunk_size:,} characters")
    summary_b.metric("Overlap", f"{chunk_overlap:,} characters")
    summary_c.metric("Strategy", splitter_strategy)
    strategy_help = {
        "Recursive": "Splits by paragraph and sentence boundaries where possible, then falls back to smaller separators.",
        "Character": "Splits text at a configured character separator and combines pieces into chunks.",
        "Token": "Uses token-based splitting when the tokenizer is available; token size and overlap are configured separately.",
        "Semantic": "Groups adjacent sentences by semantic similarity. This may take longer because it uses embeddings.",
    }
    st.info(strategy_help.get(splitter_strategy, "Splits document text into retrievable chunks."))
    st.caption("Preview the chunks cached from the most recent indexing operation. Change settings in Step 2 - Configure, then generate a preview or rebuild the index to apply them.")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Rebuild Index with Current Splitter Settings", key="rebuild_configured_index", width="stretch", disabled=not (manifest.get("documents") or manifest.get("webpages"))):
            try:
                with st.spinner("Re-splitting documents and rebuilding FAISS..."):
                    rebuilt_db, rebuilt_count, rebuilt_chunks = rebuild_uploaded_index(manifest, chunk_settings)
                if rebuilt_db is not None:
                    st.session_state.active_vector_db = rebuilt_db
                    st.session_state.active_index_label = "Uploaded PDF index"
                    st.session_state.chunk_preview = rebuilt_chunks[:100]
                    st.session_state.chunk_total = rebuilt_count
                    vector_db = rebuilt_db
                    st.success(f"Index rebuilt with {rebuilt_count} chunks using {splitter_strategy} splitting.")
                    st.rerun()
                else:
                    st.warning("No searchable content was found.")
            except Exception as exc:
                st.error("Could not rebuild the index.")
                st.exception(exc)
    with col_b:
        if st.button("Generate Chunk Preview", key="generate_chunk_preview", width="stretch", disabled=not (manifest.get("documents") or manifest.get("webpages"))):
            try:
                with st.spinner("Creating chunk preview..."):
                    preview, total = preview_chunks(manifest, chunk_settings, limit=100)
                st.session_state.chunk_preview = preview
                st.session_state.chunk_total = total
                st.success(f"Prepared a preview of {min(total, 100)} of {total} chunks.")
            except Exception as exc:
                st.error("Chunk preview failed.")
                st.exception(exc)

    if st.session_state.get("chunk_preview"):
        st.metric("Total chunks", st.session_state.get("chunk_total", len(st.session_state.chunk_preview)))
        preview_items = st.session_state.chunk_preview
        for chunk_index, chunk in enumerate(preview_items[:30], start=1):
            meta = chunk.metadata if hasattr(chunk, "metadata") else chunk.get("metadata", {})
            content = chunk.page_content if hasattr(chunk, "page_content") else chunk.get("page_content", "")
            source_label = meta.get("source", "Unknown source")
            page_label = meta.get("page_number", "-")
            with st.expander(f"Chunk {chunk_index} | {source_label} | Page {page_label} | {meta.get('source_type', 'document')}"):
                st.code(content[:4000] or "[Empty chunk]")
                st.json(meta)
        if len(preview_items) > 30:
            st.caption("Showing first 30 chunks to keep the UI responsive; the preview cache may contain up to 100.")

    st.divider()
    st.markdown("**Remove documents from the knowledge base**")
    saved_pdf_names = sorted(manifest.get("documents", {}).keys())
    saved_web_urls = sorted(manifest.get("webpages", {}).keys())
    remove_pdfs = st.multiselect("PDFs to remove", saved_pdf_names, key="remove_pdf_records")
    remove_urls = st.multiselect("Webpages to remove", saved_web_urls, key="remove_web_records")
    if st.button("Remove selected sources", key="remove_sources_button", type="secondary", disabled=not (remove_pdfs or remove_urls)):
        for name in remove_pdfs:
            manifest.get("documents", {}).pop(name, None)
        for url in remove_urls:
            manifest.get("webpages", {}).pop(url, None)
        write_manifest(manifest)
        try:
            rebuilt_db, rebuilt_count, rebuilt_chunks = rebuild_uploaded_index(manifest, chunk_settings)
            st.session_state.active_vector_db = rebuilt_db
            st.session_state.active_index_label = "Uploaded PDF index" if rebuilt_db is not None else ""
            st.session_state.chunk_preview = rebuilt_chunks[:100]
            st.session_state.chunk_total = rebuilt_count
            st.session_state.messages = []
            st.success(f"Removed selected sources. Remaining index: {rebuilt_count} chunks.")
            st.rerun()
        except Exception as exc:
            st.error("Sources were removed from the manifest, but index rebuilding failed. Retry the rebuild button.")
            st.exception(exc)

# Optional evaluation and integrity panels removed from the default interface to keep the workspace focused.

# =====================================================
# 7. DASHBOARD METRICS
# =====================================================

index_exists = vector_db is not None

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">AI AI ENGINE</div>
            <div class="metric-value">GPT-4o mini</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Search RETRIEVAL</div>
            <div class="metric-value">FAISS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    db_status = "Connected" if index_exists else "Upload PDFs"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Documents KNOWLEDGE BASE</div>
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
You are a professional research assistant for PDFs and webpages.

Answer the user's question using the supplied retrieved context, which may include PDF pages and webpages linked from those PDFs.

Instructions:
- Use the provided context as your primary source.
- Distinguish PDF evidence from webpage evidence when relevant.
- Do not invent facts, source titles, URLs, or page numbers.
- If the context does not contain the answer, say you could not find it in the indexed sources.
- Explain the answer clearly and organize it with bullets when that improves readability.

Retrieved PDF and webpage context:
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
        '<div class="section-title">Welcome Welcome to your workspace</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Start with a question or choose an example below.'
        '</div>',
        unsafe_allow_html=True,
    )

    suggestions = [
        ("Explain Explain a concept", "Explain the main concepts in my PDFs."),
        ("Summary Summarize documents", "Summarize the key points in my PDFs."),
        ("Key details Find key details", "What are the most important details in my PDFs?"),
    ]

    suggestion_cols = st.columns(3)

    for i, (label, sample_question) in enumerate(suggestions):
        with suggestion_cols[i]:
            if st.button(label, width="stretch", key=f"suggest_{i}"):
                st.session_state.pending_question = sample_question
                st.rerun()


# Display existing messages, including any PDF visuals attached to assistant replies.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for visual in message.get("visuals", []):
            image_path = Path(str(visual.get("image_path", "")))
            if image_path.is_file():
                source_label = visual.get("source", "PDF")
                page_label = visual.get("page_number", "Unknown")
                image_label = visual.get("image_number")
                caption = f"{source_label} - Page {page_label}"
                if image_label is not None:
                    caption += f" - Image {image_label}"
                st.image(str(image_path), caption=caption, width="stretch")



def is_table_display_request(question: str) -> bool:
    """Detect requests to show extracted tables from uploaded PDFs."""
    q = question.lower().strip().rstrip("?.! ")
    if q in {"table", "tables", "tabular data", "spreadsheet"}:
        return True
    display_terms = (
        "show", "display", "view", "see", "list", "extract", "find",
        "give me", "get me", "print", "export",
    )
    table_terms = ("table", "tables", "tabular data", "spreadsheet")
    if any(phrase in q for phrase in (
        "show tables", "show me tables", "show me the tables",
        "tables in this pdf", "tables in the pdf", "display tables",
        "extract tables", "all tables", "show table data",
    )):
        return True
    return any(term in q for term in display_terms) and any(term in q for term in table_terms)


def table_rows_to_records(rows):
    """Convert extracted table rows into dictionaries suitable for st.dataframe."""
    if not rows:
        return []
    headers = []
    for index, value in enumerate(rows[0]):
        name = str(value).strip() or f"Column {index + 1}"
        # Ensure dataframe column names are unique.
        original = name
        suffix = 2
        while name in headers:
            name = f"{original} ({suffix})"
            suffix += 1
        headers.append(name)
    records = []
    for row in rows[1:]:
        padded = list(row[:len(headers)]) + [""] * max(0, len(headers) - len(row))
        records.append({headers[i]: padded[i] for i in range(len(headers))})
    # If the table has only one row, still display the header as a record.
    if not records:
        records = [{headers[i]: "" for i in range(len(headers))}]
    return records


def is_image_analysis_request(question: str) -> bool:
    """Detect requests asking the vision model to analyze/summarize a PDF image."""
    q = question.lower()
    analysis_terms = (
        "analyse", "analyze", "analysis", "summarize", "summarise",
        "summary", "describe", "explain this image", "what is in this image",
        "what does this image", "interpret this image", "read this image",
    )
    image_terms = ("image", "picture", "photo", "visual", "chart", "diagram", "figure")
    has_analysis_intent = any(term in q for term in analysis_terms)
    has_visual_target = any(term in q for term in image_terms)
    return has_analysis_intent and has_visual_target


def is_image_display_request(question: str) -> bool:
    """Detect requests to display PDF page previews rather than answer from text."""
    q = question.lower()
    display_terms = (
        "show", "display", "view", "see", "list", "find", "extract",
        "give me", "get me", "where are",
    )
    visual_terms = (
        "image", "images", "picture", "pictures", "photo", "photos",
        "visual", "visuals", "chart", "charts", "diagram", "diagrams",
        "scanned page", "scanned pages",
    )
    return (
        any(term in q for term in display_terms)
        and any(term in q for term in visual_terms)
    ) or any(phrase in q for phrase in (
        "images in this pdf", "images in the pdf", "all pdf images",
        "all images", "show pdf images", "show me the images",
        "show me images", "show charts", "show diagrams",
    ))


# =====================================================
# 10. HANDLE USER QUESTION
# =====================================================

question = st.chat_input(
    "Ask anything about your PDF documents...",
    disabled=(vector_db is None and not st.session_state.get("pdf_visual_pages") and not st.session_state.get("pdf_embedded_images") and not st.session_state.get("pdf_extracted_tables")),
)

if vector_db is None and not st.session_state.get("pdf_visual_pages") and not st.session_state.get("pdf_embedded_images") and not st.session_state.get("pdf_extracted_tables"):
    st.info("Upload one or more PDFs in the sidebar and click **Process uploaded PDFs** to create your searchable knowledge base.")

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
            with st.spinner("Checking your PDF visuals and preparing the answer..."):
                related_visuals = []
                related_visuals = filter_visuals_for_question(question, related_visuals)
                sources = []

                # URL-count questions should use the PDF extraction manifest directly,
                # not vector retrieval (and must never display unrelated PDF images).
                if is_url_count_request(question):
                    current_manifest = read_manifest()
                    urls_by_pdf = {}
                    for pdf_name, record in current_manifest.get("documents", {}).items():
                        for url in record.get("urls", []):
                            if isinstance(url, str) and url.strip():
                                urls_by_pdf.setdefault(url.strip(), set()).add(pdf_name)
                    url_list = sorted(urls_by_pdf)
                    answer = f"There are **{len(url_list)} unique URL(s)** found in the processed PDF(s)."
                    if url_list:
                        answer += "\n\n" + "\n".join(f"- {url}" for url in url_list)
                        sources.append("PDF URL extraction manifest")
                    else:
                        answer = (
                            "I couldn't find any extracted URLs in the processed PDF records. "
                            "Reprocess the PDF if it was uploaded before URL extraction was enabled."
                        )
                # Table-display requests use extracted structured tables directly;
                # FAISS text similarity alone may not retrieve every table.
                elif is_table_display_request(question):
                    saved_tables = st.session_state.get("pdf_extracted_tables", [])
                    if saved_tables:
                        answer = f"Found **{len(saved_tables)} extracted table(s)** in the uploaded PDFs."
                        for table in saved_tables:
                            sources.append(
                                f"**{table['source']}** - Page {table['page_number']}, Table {table['table_number']}"
                            )
                    else:
                        answer = (
                            "I couldn't detect any structured tables in the PDFs processed in this session. "
                            "If a table is a scanned image, it may require OCR/layout recognition; "
                            "please re-upload and process the PDF, or use the page preview to inspect it."
                        )
                # Image-analysis requests send the first saved PDF page preview to the
                # vision-capable model, rather than asking text-only FAISS to describe it.
                elif is_image_analysis_request(question):
                    saved_images = st.session_state.get("pdf_embedded_images", [])
                    saved_visuals = st.session_state.get("pdf_visual_pages", [])
                    valid_images = [v for v in saved_images if Path(v.get("image_path", "")).is_file()]
                    valid_previews = [v for v in saved_visuals if Path(v.get("image_path", "")).is_file()]
                    image_number_match = re.search(r"\b(?:image|picture|photo)\s*(?:number\s*)?(\d+)\b", question.lower())
                    if image_number_match:
                        requested_image_index = int(image_number_match.group(1)) - 1
                        first_visual = valid_images[requested_image_index] if 0 <= requested_image_index < len(valid_images) else None
                    elif any(term in question.lower() for term in ("first image", "first picture", "first chart", "first diagram")):
                        first_visual = valid_images[0] if valid_images else None
                    else:
                        first_visual = valid_images[0] if valid_images else None
                    if first_visual is None:
                        first_visual = valid_previews[0] if valid_previews else None
                    if first_visual is None:
                        answer = (
                            "I don't have a saved image preview to analyze yet. Upload the PDF "
                            "and click **Process uploaded PDFs** first. The loader currently saves "
                            "page previews for pages with embedded images or little selectable text."
                        )
                    else:
                        image_path = Path(first_visual["image_path"])
                        image_data_url = image_file_to_data_url(image_path)
                        vision_message = HumanMessage(content=[
                            {
                                "type": "text",
                                "text": (
                                    "Analyze the attached first extracted PDF image (or page preview if no individual image was extractable). Give a concise, "
                                    "accurate summary of visible content, including the main subject, "
                                    "important labels or values that are legible, and any chart/diagram "
                                    "meaning that can be reasonably inferred. Do not guess unreadable details. "
                                    f"User request: {question}"
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ])
                        response = llm.invoke([vision_message])
                        answer = str(response.content)
                        related_visuals.append({
                            "source": first_visual.get("source", "Unknown PDF"),
                            "page_number": first_visual.get("page_number", "Unknown"),
                            "image_path": str(image_path),
                        })
                        sources.append(
                            f"**{first_visual.get('source', 'Unknown PDF')}** - "
                            f"Page {first_visual.get('page_number', 'Unknown')} (first extracted image or page preview)"
                        )
                # Image-display requests bypass text similarity search.
                # FAISS can miss visual pages when those pages contain little text.
                elif is_image_display_request(question):
                    saved_images = st.session_state.get("pdf_embedded_images", [])
                    saved_visuals = st.session_state.get("pdf_visual_pages", [])
                    q_lower = question.lower()

                    requested_number = None
                    number_match = re.search(r"\b(?:image|picture|photo)\s*(?:number\s*)?(\d+)\b", q_lower)
                    if number_match:
                        requested_number = int(number_match.group(1))
                    elif any(term in q_lower for term in ("first image", "first picture", "first photo", "first chart", "first diagram")):
                        requested_number = 1

                    page_match = re.search(r"\bpage\s*(\d+)\b", q_lower)
                    requested_page = int(page_match.group(1)) if page_match else None

                    candidates = [v for v in saved_images if Path(v.get("image_path", "")).is_file()]
                    if requested_page is not None:
                        candidates = [v for v in candidates if v.get("page_number") == requested_page]

                    if requested_number is not None:
                        chosen = candidates[requested_number - 1:requested_number]
                    else:
                        chosen = candidates

                    # Scanned PDFs may not expose separate image objects; use page previews then.
                    if not chosen and not candidates:
                        preview_candidates = [v for v in saved_visuals if Path(v.get("image_path", "")).is_file()]
                        if requested_page is not None:
                            preview_candidates = [v for v in preview_candidates if v.get("page_number") == requested_page]
                        chosen = preview_candidates[requested_number - 1:requested_number] if requested_number is not None else preview_candidates

                    for visual in chosen:
                        related_visuals.append({
                            "source": visual.get("source", "Unknown PDF"),
                            "page_number": visual.get("page_number", "Unknown"),
                            "image_number": visual.get("image_number", visual.get("page_image_number")),
                            "image_path": visual["image_path"],
                        })
                        sources.append(
                            f"**{visual.get('source', 'Unknown PDF')}** - Page {visual.get('page_number', 'Unknown')}"
                        )

                    if related_visuals:
                        if requested_number is not None:
                            answer = f"Showing image **{requested_number}** from the available PDF visuals."
                        elif requested_page is not None:
                            answer = f"Showing **{len(related_visuals)} image(s)** from PDF page **{requested_page}**."
                        elif saved_images:
                            answer = f"Found **{len(related_visuals)} individual embedded image(s)**. Ask `show image 1`, `show image 2`, or `show images from page 3` to narrow the selection."
                        else:
                            answer = f"No individual image objects were extractable, so I'm showing **{len(related_visuals)} page preview(s)** instead."
                    else:
                        answer = (
                            "I couldn't find a matching image or page preview. Re-upload the PDF and click "
                            "**Process uploaded PDFs**. Try `show image 1` or `show images from page 2`."
                        )
                else:
                    candidates = hybrid_retrieve(
                        question, vector_db, k=40, use_hybrid=use_hybrid_search
                    ) if vector_db is not None else []

                    def apply_retrieval_filters(candidate_docs):
                        filtered = []
                        for doc in candidate_docs:
                            meta = doc.metadata or {}
                            source = str(meta.get("source", ""))
                            source_type = str(meta.get("source_type", "pdf"))
                            try:
                                page_no = int(meta.get("page_number", 0) or 0)
                            except (TypeError, ValueError):
                                page_no = 0
                            if selected_pdf_filter and source not in selected_pdf_filter:
                                continue
                            if document_type_filter == "PDF pages" and source_type != "pdf":
                                continue
                            if document_type_filter == "Tables" and source_type != "table":
                                continue
                            if document_type_filter == "Webpages" and source_type != "webpage":
                                continue
                            if document_type_filter == "Other documents" and source_type in ("pdf", "table", "webpage"):
                                continue
                            if source_type in ("pdf", "table"):
                                if page_no < int(page_start):
                                    continue
                                if int(page_end) > 0 and page_no > int(page_end):
                                    continue
                            if selected_url_filter and source_type == "webpage" and source not in selected_url_filter:
                                continue
                            if selected_url_filter and source_type != "webpage":
                                continue
                            filtered.append(doc)
                        return filtered

                    candidates = apply_retrieval_filters(candidates)
                    relevance_score = None
                    corrective_used = False

                    if candidates and use_corrective_rag:
                        documents, relevance_score, corrective_used = corrective_retrieve(
                            llm, question, vector_db, candidates,
                            top_n=rerank_top_n,
                            threshold=float(corrective_threshold),
                            use_hybrid=use_hybrid_search,
                        )
                        documents = apply_retrieval_filters(documents)
                        # If source/page filters remove the corrected candidates, rerank the
                        # filtered initial set rather than accidentally answering from other sources.
                        if not documents and candidates:
                            documents, relevance_score = rerank_documents(
                                llm, question, candidates, top_n=rerank_top_n
                            )
                    elif candidates and use_reranking:
                        documents, relevance_score = rerank_documents(
                            llm, question, candidates, top_n=rerank_top_n
                        )
                    else:
                        documents = candidates[:rerank_top_n]

                    if documents and relevance_score is not None:
                        st.caption(
                            f"Retrieval: {'Hybrid BM25 + FAISS' if use_hybrid_search else 'FAISS only'}"
                            f" · Reranking {'on' if (use_reranking or use_corrective_rag) else 'off'}"
                            f" · Best relevance score: {relevance_score:.0f}/100"
                            f" · Corrective retry: {'used' if corrective_used else 'not needed'}"
                        )

                    if not documents:
                        answer = (
                            "I could not find relevant text matching that question and the current metadata filters. "
                            "Try clearing the PDF, page, document-type, or URL filters. You can also inspect the chunk preview to confirm what was indexed."
                        )
                    else:
                        context = "\\n\\n".join(
                            doc.page_content for doc in documents
                        )
                        messages = prompt.format_messages(
                            context=context,
                            question=question,
                        )
                        response = llm.invoke(messages)
                        answer = response.content

                        seen_visual_paths = set()
                        for doc in documents:
                            filename = Path(
                                doc.metadata.get("source", "Unknown")
                            ).name
                            page_number = doc.metadata.get("page_number")
                            page = doc.metadata.get("page")
                            if isinstance(page_number, int):
                                page_display = page_number
                            elif isinstance(page, int):
                                page_display = page + 1
                            else:
                                page_display = "Unknown"

                            if doc.metadata.get("source_type") == "webpage":
                                sources.append(f"**Webpage:** [{doc.metadata.get('title', filename)}]({doc.metadata.get('url', filename)})")
                            else:
                                sources.append(f"**{filename}** - Page {page_display}")
                            # Do not attach page images to ordinary text/URL answers.
                            # Images are rendered only by explicit image-display/analysis paths.
                            if classify_answer_intent(question) == "images":
                                visual_path = doc.metadata.get("page_image")
                                if visual_path and visual_path not in seen_visual_paths:
                                    seen_visual_paths.add(visual_path)
                                    related_visuals.append({
                                        "source": filename,
                                        "page_number": page_display,
                                        "image_path": visual_path,
                                    })

            st.markdown(answer)

            # Render the selected PDF images in the same assistant response.
            # Previously only a text confirmation was shown, while the image paths
            # were stored in message metadata but never passed to st.image().
            for visual in (related_visuals if classify_answer_intent(question) == "images" else []):
                image_path = Path(str(visual.get("image_path", "")))
                if image_path.is_file():
                    source_label = visual.get("source", "PDF")
                    page_label = visual.get("page_number", "Unknown")
                    image_label = visual.get("image_number")
                    caption = f"{source_label} - Page {page_label}"
                    if image_label is not None:
                        caption += f" - Image {image_label}"
                    st.image(str(image_path), caption=caption, width="stretch")
                else:
                    st.warning("The selected PDF image file is missing. Reprocess the uploaded PDF to regenerate it.")

            if is_table_display_request(question):
                saved_tables = st.session_state.get("pdf_extracted_tables", [])
                for table in saved_tables:
                    st.markdown(
                        f"**{table['source']} - Page {table['page_number']} - Table {table['table_number']}**"
                    )
                    records = table_rows_to_records(table.get("rows", []))
                    if records:
                        st.dataframe(records, width="stretch", hide_index=True)
                    else:
                        st.caption("This detected table did not contain displayable rows.")


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "visuals": related_visuals,
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
        MultiPDF AI Assistant | Built with Streamlit, LangChain,
        FAISS & OpenAI
        <br>
        Answers use retrieved text; image descriptions are optional and generated with vision AI.
        Verify important information against the original PDFs.
    </div>
    """,
    unsafe_allow_html=True,
)
