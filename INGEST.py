# ingest.py
import os
import re
from typing import List

import fitz  # PyMuPDF
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def parse_iso14229_pdf(pdf_path: str) -> List[Document]:
    """
    Parse ISO 14229-1:2013 PDF and extract page-specific documents,
    annotated with service metadata.
    Only first 5 services are kept (ignoring index pages).
    """
    print(f"Parsing ISO14229 PDF: {pdf_path}")

    # Regex captures section, service name, and service ID
    service_patterns = [
        r'(\d{1,2}\.\d{1,2})\s+([\w\d]+)\s+\(0x([0-9A-Fa-f]+)\)\s+service'
    ]

    page_docs: List[Document] = []
    service_index = {}

    # Step 1: Use PyMuPDF to detect service start pages and define page ranges
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        text = doc.load_page(page_num).get_text()
        text = re.sub(r'\s+', ' ', text)  # normalize whitespace

        for pattern in service_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                section_num, service_name, service_id_hex = match
                service_id = f"0x{service_id_hex}"

                # Overwrite ensures we skip index entries and keep actual definition later
                service_index[service_name] = {
                    "service_name": service_name,
                    "service_id": service_id,
                    "section": section_num,
                    "start": page_num
                }

    # Convert dict → sorted list by page number
    page_ranges = list(service_index.values())
    page_ranges = sorted(page_ranges, key=lambda x: x["start"])

    # Limit to first 5 services
    page_ranges = page_ranges[:5]

    # Add end markers
    for i in range(len(page_ranges)):
        if i < len(page_ranges) - 1:
            page_ranges[i]["end"] = page_ranges[i + 1]["start"]
        else:
            page_ranges[i]["end"] = len(doc)
    doc.close()

    # Step 2: Extract content page by page using PDFPlumber
    with pdfplumber.open(pdf_path) as pdf:
        for info in page_ranges:
            # --- This metadata applies to all pages in this range ---
            service_metadata = {
                "service_name": info["service_name"],
                "service_id": info["service_id"],
                "section": info["section"]
            }

            for page_num in range(info["start"], info["end"]):
                if page_num >= len(pdf.pages):
                    continue

                page = pdf.pages[page_num]
                page_text = page.extract_text() or ""

                # Extract tables from the page
                tables = page.extract_tables()
                tables_text_list = []
                for table in tables:
                    cleaned_rows = [
                        " | ".join(map(lambda cell: '' if cell is None else cell, row))
                        for row in table
                    ]
                    tables_text_list.append("\n".join(cleaned_rows))

                tables_text = "\n\n".join(tables_text_list)

                # Combine text and tables for the current page
                page_content = f"## Page Text\n{page_text.strip()}\n\n## Page Tables\n{tables_text}"

                # --- Create a Document for EACH page ---
                page_metadata = service_metadata.copy()
                page_metadata["page"] = page_num + 1  # human-friendly numbering

                page_doc = Document(
                    page_content=page_content,
                    metadata=page_metadata
                )
                page_docs.append(page_doc)

    print(f"Extracted {len(page_docs)} page-level documents from the PDF.")
    return page_docs


# --- MAIN EXECUTION LOGIC ---
if __name__ == "__main__":
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    pdf_path = "ISO_14229-1_2013.en.PDF (1).pdf"  # <-- Set your PDF path
    collection_name = "iso14229_uds_pages"

    # 1. Parse the PDF to get page-level documents
    page_level_docs = parse_iso14229_pdf(pdf_path)

    if not page_level_docs:
        print("No documents were extracted.")
    else:
        # 2. Split the page-level documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        final_chunks = text_splitter.split_documents(page_level_docs)
        print(f"Created a total of {len(final_chunks)} chunks for embedding.")

        # 3. Initialize Gemini Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # 4. Upload to Qdrant
        print("Uploading document chunks to Qdrant...")
        Qdrant.from_documents(
            final_chunks,
            embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            force_recreate=True,
        )
        print(f"Successfully uploaded {len(final_chunks)} chunks to collection '{collection_name}'.")
