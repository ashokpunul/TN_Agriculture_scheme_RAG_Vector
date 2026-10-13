
import logging

import streamlit as st

from pdf_loader import load_uploaded_pdfs


logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Professional PDF Loader",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Professional PDF Loader")
st.caption(
    "Validate documents, inspect page quality, and detect image/OCR candidates."
)

uploads = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Inspect PDFs", type="primary"):
    if not uploads:
        st.warning("Upload at least one PDF.")
    else:
        with st.spinner("Validating and inspecting PDFs..."):
            documents, reports = load_uploaded_pdfs(uploads)

        st.subheader("Ingestion summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Extracted pages", len(documents))
        col2.metric("Uploaded files", len(uploads))
        col3.metric(
            "OCR candidate pages",
            sum(
                len(r.get("ocr_candidate_pages", []))
                for r in reports
            ),
        )

        st.subheader("File reports")

        for report in reports:
            with st.expander(
                f"{report['filename']} — {report['status']}"
            ):
                st.json(report)

        st.subheader("Extracted text preview")

        for doc in documents[:5]:
            st.markdown(
                f"**{doc.metadata['source']} — "
                f"Page {doc.metadata['page_number']}**"
            )
            st.text(doc.page_content[:1000])

        st.success(
            f"Inspection finished. {len(documents)} "
            "page documents are ready for the next layer."
        )
