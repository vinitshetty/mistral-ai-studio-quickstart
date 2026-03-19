import asyncio
import sys
from pathlib import Path

# Add project root to sys.path for cross-folder imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workflows.workflow.worker import process_document_ocr, extract_invoice_data, DocumentInput

INVOICES_DIR = Path(__file__).resolve().parents[2] / "invoices"

async def main():
    """Read a local invoice and process it"""
    document_path = str(sorted(INVOICES_DIR.glob("*.jpg"))[0])
    print(f"Processing: {document_path}")

    # Process the document with OCR
    ocr_result = await process_document_ocr(DocumentInput(document_path=document_path))
    invoice_data = await extract_invoice_data(ocr_result)
    print("OCR Processing Complete!")
    print("Raw Text:", ocr_result.raw_text[:500])
    print("Structured Output result:", invoice_data)

if __name__ == "__main__":
    asyncio.run(main())
    