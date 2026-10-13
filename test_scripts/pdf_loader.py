
# pdf_loader.py

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_core.documents import Document


import io
import logging

import fitz
import pytesseract
from PIL import Image



logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 50
MAX_PAGES = 2000
MAX_TEXT_CHARS_PER_PAGE = 2_000_000
MIN_TEXT_CHARS = 30


def sha256_bytes(data: bytes) -> str:
    """Create a stable fingerprint of the original PDF bytes."""
    return hashlib.sha256(data).hexdigest()


def inspect_page(page: fitz.Page) -> dict[str, Any]:
    """Inspect text and embedded-image indicators for one page."""

    text = page.get_text("text", sort=True).strip()
    images = page.get_images(full=True)

    # This is a heuristic, not a definitive OCR requirement.
    likely_scanned = len(text) < MIN_TEXT_CHARS

    return {
        "page_number": page.number + 1,
        "text": text,
        "text_characters": len(text),
        "embedded_image_count": len(images),
        "page_width": round(page.rect.width, 2),
        "page_height": round(page.rect.height, 2),
        "likely_needs_ocr": likely_scanned,
        "quality_status": (
            "LOW_TEXT"
            if likely_scanned
            else "TEXT_EXTRACTED"
        ),
    }


def load_pdf_bytes(
    filename: str,
    pdf_bytes: bytes,
) -> tuple[list[Document], dict[str, Any]]:
    """
    Validate and inspect a PDF.

    Returns:
        Documents: One LangChain Document per usable page.
        Report: File-level and page-level diagnostics.

    Does not perform OCR or extract image files.
    """

    documents: list[Document] = []
    report: dict[str, Any] = {
        "filename": Path(filename).name,
        "status": "PENDING",
        "pages": 0,
        "text_characters": 0,
        "embedded_images": 0,
        "ocr_candidate_pages": [],
        "warnings": [],
        "errors": [],
        "page_details": [],
    }

    safe_filename = Path(filename).name

    # 1. Basic file validation
    if not pdf_bytes:
        report["status"] = "REJECTED"
        report["errors"].append("File is empty.")
        return documents, report

    if len(pdf_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        report["status"] = "REJECTED"
        report["errors"].append(
            f"File exceeds {MAX_FILE_SIZE_MB} MB."
        )
        return documents, report

    if not pdf_bytes.startswith(b"%PDF-"):
        report["status"] = "REJECTED"
        report["errors"].append("Invalid PDF signature.")
        return documents, report

    file_hash = sha256_bytes(pdf_bytes)
    report["sha256"] = file_hash
    report["file_size_bytes"] = len(pdf_bytes)

    # 2. Parse and inspect the PDF
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:

            if pdf.needs_pass:
                report["status"] = "REJECTED"
                report["errors"].append(
                    "Password-protected PDF is not supported."
                )
                return documents, report

            if pdf.page_count == 0:
                report["status"] = "REJECTED"
                report["errors"].append("PDF has no pages.")
                return documents, report

            if pdf.page_count > MAX_PAGES:
                report["status"] = "REJECTED"
                report["errors"].append(
                    f"PDF exceeds the {MAX_PAGES}-page limit."
                )
                return documents, report

            # Capture basic document metadata.
            pdf_metadata = pdf.metadata or {}
            title = pdf_metadata.get("title") or safe_filename
            author = pdf_metadata.get("author") or "Unknown"

            report["title"] = title
            report["author"] = author
            report["pages"] = pdf.page_count

            for page in pdf:
                page_info = inspect_page(page)

                if (
                    page_info["text_characters"]
                    > MAX_TEXT_CHARS_PER_PAGE
                ):
                    report["warnings"].append(
                        f"Page {page.number + 1}: "
                        "unusually large extracted text."
                    )
                    page_info["text"] = page_info["text"][
                        :MAX_TEXT_CHARS_PER_PAGE
                    ]
                    page_info["text_characters"] = len(
                        page_info["text"]
                    )

                text = page_info.pop("text")
                page_number = page_info["page_number"]

                report["text_characters"] += len(text)
                report["embedded_images"] += (
                    page_info["embedded_image_count"]
                )

                if page_info["likely_needs_ocr"]:
                    report["ocr_candidate_pages"].append(page_number)

                report["page_details"].append(page_info)

                # Skip empty pages from the vector-search documents,
                # but preserve their diagnostics in the report.
                if not text.strip():
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": safe_filename,
                            "source_type": "pdf",
                            "file_sha256": file_hash,
                            "page": page_number - 1,
                            "page_number": page_number,
                            "title": title,
                            "author": author,
                            "extraction_method": "pymupdf_text",
                            "embedded_image_count": (
                                page_info["embedded_image_count"]
                            ),
                            "likely_needs_ocr": (
                                page_info["likely_needs_ocr"]
                            ),
                        },
                    )
                )

        # 3. Determine the overall extraction status.
        if not documents:
            report["status"] = "NEEDS_OCR_OR_REVIEW"
            report["warnings"].append(
                "No usable selectable text was extracted."
            )
        elif report["ocr_candidate_pages"]:
            report["status"] = "LOADED_WITH_OCR_CANDIDATES"
            report["warnings"].append(
                "Some pages have little text; inspect them for scans."
            )
        else:
            report["status"] = "LOADED"

    except Exception as exc:
        logger.exception("Failed to parse PDF: %s", safe_filename)
        report["status"] = "FAILED"
        report["errors"].append(
            f"{type(exc).__name__}: {exc}"
        )
        documents = []

    return documents, report


def load_uploaded_pdfs(
    uploaded_files,
) -> tuple[list[Document], list[dict[str, Any]]]:
    """Load multiple Streamlit uploads with duplicate detection."""

    all_documents: list[Document] = []
    reports: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name

        try:
            data = uploaded_file.getvalue()
            fingerprint = sha256_bytes(data) if data else None

            if fingerprint and fingerprint in seen_hashes:
                reports.append({
                    "filename": filename,
                    "status": "SKIPPED_DUPLICATE",
                    "sha256": fingerprint,
                    "errors": [],
                    "warnings": [
                        "Identical file content was already uploaded."
                    ],
                    "page_details": [],
                })
                continue

            if fingerprint:
                seen_hashes.add(fingerprint)

            docs, report = load_pdf_bytes(filename, data)
            reports.append(report)
            all_documents.extend(docs)

        except Exception as exc:
            logger.exception("Upload failed: %s", filename)
            reports.append({
                "filename": filename,
                "status": "FAILED",
                "errors": [
                    f"{type(exc).__name__}: {exc}"
                ],
                "warnings": [],
                "page_details": [],
            })

    return all_documents, reports
