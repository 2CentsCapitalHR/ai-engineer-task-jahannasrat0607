[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vgbm4cZ0)

# ADGM Corporate Agent — Document Intelligence

An AI-powered legal document review system for Abu Dhabi Global Market (ADGM) corporate compliance. This tool helps automate the review of company formation and related documents against ADGM regulations using OpenAI’s LLMs combined with a FAISS-based retrieval-augmented generation (RAG) pipeline.

---

## Project Overview

This project enables users (corporate agents, legal teams, compliance officers) to upload '.docx' documents related to company incorporation, board resolutions, shareholder agreements, UBO declarations, and other corporate filings. The system:

- Automatically detects the document type and associated ADGM process
- Runs rule-based checks for common compliance issues
- Performs an advanced RAG-based analysis leveraging a FAISS vector store of ADGM regulatory texts
- Generates detailed annotated `.docx` files with in-line comments highlighting issues
- Produces JSON reports summarizing findings and checklist completeness

---

## Features

- **Multi-document upload:** Upload multiple '.docx' files to review entire incorporation packs or compliance sets.
- **Document classification:** Classifies files by keywords to identify types such as Articles of Association, UBO Declarations, Board Resolutions, etc.
- **Process detection & checklist:** Determines the corporate process and identifies any missing required documents.
- **Rule-based compliance checks:** Detects missing jurisdiction clauses, signatures, dates, ambiguous language, etc.
- **RAG-backed AI review:** Uses OpenAI LLMs combined with a FAISS vector store for in-depth paragraph-level compliance review.
- **Annotated documents:** Outputs reviewed '.docx' files with inline comments and a summary comments page.
- **Comprehensive JSON reports:** Provides structured results for integration or further analysis.
- **Streamlit UI:** Intuitive web interface for uploading, running reviews, viewing results, and downloading outputs.

---

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key with access to GPT-4o-mini or GPT-4 models
- Installed dependencies ('requirements.txt')

### Installation

1. Clone this repository:
   git clone https://github.com/2CentsCapitalHR/ai-engineer-task-jahannasrat0607
   cd adgm-corporate-agent
2. install dependencies
   pip install -r requirements.txt
3. Set environment variables in .env:
    OPENAI_API_KEY=your_openai_api_key
    FAISS_DIR=path_to_your_faiss_store
    LLM_MODEL_NAME=gpt-4o-mini
4. Prepare your FAISS vector store under the folder specified in FAISS_DIR. (Refer to project docs for building the vector store from ADGM regulatory texts.)

### Running the Streamlit App

streamlit run streamlit_app.py
Then open your browser at http://localhost:8501

---

### Project Structure
- backend.py: Core logic for document text extraction, classification, FAISS & LLM loading, rule-based checks, RAG review, and document annotation.
- streamlit_app.py: Streamlit user interface for file upload, triggering reviews, and displaying annotated results.
- faiss_store/: Directory containing FAISS index and embeddings.
- reports/: Output folder for reviewed .docx files and JSON reports.
- .env: Environment variables for API keys and config.

---

### Usage
- Upload one or more .docx legal documents related to ADGM corporate filings.
- Click Run Review to perform compliance checks.
- View detected process, checklist status, and detailed issues.
- Download annotated documents with inline comments.
- Download JSON reports for integration or audit trails.

---

### Contributing
Contributions are welcome! Please open issues or submit pull requests for enhancements, bug fixes, or additional document/process support.

---
### Contact
For questions or collaboration, please contact:

Nasrat Jahan
Email: nasratjahan166@gmail.com
LinkedIn: https://www.linkedin.com/in/nasrat-jahan-95aa76326

---

### Acknowledgements
OpenAI for LLM API
FAISS by Facebook AI for vector search
LangChain for easy LLM & retriever integrations
Python-docx for document manipulation
Streamlit for UI framework

