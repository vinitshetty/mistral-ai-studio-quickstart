# run_local.py (updated for local execution)
import asyncio
import json
import sys
from pathlib import Path

# Add project root to sys.path for cross-folder imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from mistralai_workflows import (
    execute_workflow,  # Use execute_workflow instead of WorkflowsClient
)
from pydantic import BaseModel

from workflows.workflow.worker import OCRDocumentWorkflow

load_dotenv()

INVOICES_DIR = Path(__file__).resolve().parents[2] / "invoices"

class OCRWorkflowInput(BaseModel):
    document_path: str

async def main() -> None:
    document_path = str(sorted(INVOICES_DIR.glob("*.jpg"))[0])

    print("=" * 70)
    print("OCR INVOICE WORKFLOW (LOCAL EXECUTION)")
    print("=" * 70)
    print(f"\nDocument: {document_path}\n")

    # Execute the workflow locally (no Temporal server)
    result = await execute_workflow(
        workflow=OCRDocumentWorkflow,  # Your workflow class
        params=OCRWorkflowInput(document_path=document_path),  # Input as Pydantic model
    )

    print("Workflow completed!")
    print("=" * 70)

    # Convert Pydantic model to dict before JSON serialization
    result_dict = result.model_dump() if hasattr(result, "model_dump") else result

    if isinstance(result_dict, dict):
        decision = result_dict.get("decision", "unknown")
        total_amount = result_dict.get("total_amount", 0)
        required_approval = result_dict.get("required_human_approval", False)
        invoice_category = result_dict.get("invoice_category", "")

        print(f"\nDECISION: {decision}")
        print(f"Total Amount: {total_amount} EUR")
        print(f"Threshold: {100.0} EUR")
        print(f"Required Human Approval: {required_approval}")
        print(f"Invoice category: {invoice_category}")
        print("-" * 70)

        print("\nEXTRACTED DATA:")
        print("-" * 70)
        extracted_data = result_dict.get("extracted_data", {})
        print(json.dumps(extracted_data, indent=2, ensure_ascii=False))
    else:
        # Handle case where result is a Pydantic model
        decision = result.decision if hasattr(result, "decision") else "unknown"
        total_amount = result.total_amount if hasattr(result, "total_amount") else 0
        required_approval = result.required_human_approval if hasattr(result, "required_human_approval") else False
        invoice_category = result.invoice_category if hasattr(result, "invoice_category") else ""

        print(f"\nDECISION: {decision}")
        print(f"Total Amount: {total_amount} EUR")
        print(f"Threshold: {100.0} EUR")
        print(f"Required Human Approval: {required_approval}")
        print(f"Invoice Category: {invoice_category}")
        print("-" * 70)

        print("\nEXTRACTED DATA:")
        print("-" * 70)
        extracted_data = result.extracted_data if hasattr(result, "extracted_data") else {}
        print(json.dumps(extracted_data, indent=2, ensure_ascii=False))

    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
