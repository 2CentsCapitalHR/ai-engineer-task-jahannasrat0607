# app.py
import os
import tempfile
import json
import streamlit as st  # type: ignore
from dotenv import load_dotenv
from docx import Document

# try importing backend
try:
    from backend import process_documents
except Exception as e:
    process_documents = None
    backend_import_error = e

load_dotenv()

st.set_page_config("ADGM Corporate Agent", layout="wide")
st.title("🏛️ ADGM Corporate Agent — Document Intelligence")

st.markdown(
    """
Upload your '.docx' documents (company formation, resolutions, employment contracts, licensing docs, etc.).
The system will:
- Detect which process you're trying (e.g., Company Incorporation),
- Verify checklist completeness,
- Run RAG-backed red-flag detection against ADGM sources,
- Return annotated `.docx` files and JSON reports.
"""
)

st.caption("Tip: Upload multiple docs (AoA, MoA, UBO, etc.) to test the checklist feature.")

uploaded_files = st.file_uploader("Upload .docx files", type=["docx"], accept_multiple_files=True)

# show helpful message if backend is missing
if process_documents is None:
    st.error(
        "backend.py with process_documents() was not found or failed to import. "
        "Please ensure backend.py exists and exports process_documents(file_paths)."
    )
    st.exception(backend_import_error)

def extract_text_preview(path: str, max_chars: int = 4000) -> str:
    """Return a short text preview from a .docx file."""
    try:
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        if len(full_text) > max_chars:
            return full_text[:max_chars] + "\n\n...[truncated]"
        return full_text
    except Exception as e:
        return f"[Could not read file: {e}]"

if st.button("Run Review") and uploaded_files and process_documents:
    # save uploaded files to temp dir
    tmpdir = tempfile.mkdtemp(prefix="adgm_")
    local_paths = []
    for f in uploaded_files:
        save_path = os.path.join(tmpdir, f.name)
        with open(save_path, "wb") as out:
            out.write(f.getbuffer())
        local_paths.append(save_path)

    # show Before previews
    st.subheader("Before Review (Raw content previews)")
    for p in local_paths:
        fn = os.path.basename(p)
        with st.expander(f"Before: {fn}", expanded=False):
            st.text(extract_text_preview(p))

    # process them (this loads FAISS and LLM)
    with st.spinner("Running RAG review..."):
        try:
            results = process_documents(local_paths)  # backend handles FAISS/LMM load
        except Exception as e:
            st.error(f"Processing failed: {e}")
            raise

    st.success("Processing complete")

    # show detected process and checklist
    st.subheader("Detected Process & Checklist")
    st.json(results.get("checklist", {}))

    # issues per document
    st.subheader("Issues Found (per document)")
    for r in results.get("reports", []):
        st.markdown(f"### {r.get('document')} — {r.get('doc_type')}")
        issues = r.get("issues", [])
        if not issues:
            st.info("No issues found.")
            continue
        for idx, i in enumerate(issues, start=1):
            st.write(f"**{idx}. {i.get('issue')}**  — *{i.get('severity','Medium')}*")
            if i.get("suggestion"):
                st.write(f"  - Suggestion: {i.get('suggestion')}")
            if i.get("citation"):
                st.write(f"  - Citation: {i.get('citation')}")
            # show paragraph index if available
            if i.get("para_idx") is not None:
                st.write(f"  - Paragraph index: {i.get('para_idx')}")

    # show After-preview: small extracted snippet from reviewed files if available
    st.subheader("After Review (reviewed file previews)")
    for p in results.get("reviewed_files", []):
        if not os.path.exists(p):
            continue
        fn = os.path.basename(p)
        with st.expander(f"After (annotated): {fn}", expanded=False):
            st.text(extract_text_preview(p))

    # download reviewed files
    st.subheader("Download Reviewed Documents & Reports")
    for p in results.get("reviewed_files", []):
        if os.path.exists(p):
            with open(p, "rb") as f:
                st.download_button(
                    label=f"Download {os.path.basename(p)}",
                    data=f,
                    file_name=os.path.basename(p),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

    # aggregate JSON
    agg_json = json.dumps(results, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download aggregate_report.json",
        data=agg_json.encode("utf-8"),
        file_name="aggregate_report.json",
        mime="application/json",
    )

else:
    if not uploaded_files:
        st.info("Upload one or more .docx files to enable the Run Review button.")
    elif process_documents is None:
        st.warning("Backend not available; fix import above to run the review.")