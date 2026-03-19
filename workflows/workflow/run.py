"""Execute the OCR Invoice Workflow."""

import asyncio
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from mistralai_workflows import WorkflowsClient
from pydantic import BaseModel

load_dotenv()

INVOICES_DIR = Path(__file__).resolve().parents[2] / "invoices"


class OCRWorkflowInput(BaseModel):
    """Input for the OCR workflow."""
    document_path: str


async def run_single(client: WorkflowsClient, invoice_path: Path) -> None:
    """Run the OCR workflow on a single invoice."""
    execution_id = uuid.uuid4().hex

    print(f"[{invoice_path.name}] Starting workflow for {invoice_path}")
    print(f"[{invoice_path.name}] Execution ID: {execution_id}")

    await client.execute_workflow(
        workflow_identifier="ocr_invoice_workflow_test",
        input_data=OCRWorkflowInput(document_path=str(invoice_path)),
        execution_id=execution_id,
    )

    print(f"[{invoice_path.name}] Workflow started. Waiting for completion...")
    print(f"[{invoice_path.name}] To approve: uv run python workflows/utils/approve.py {execution_id}")

    response = await client.wait_for_workflow_completion(execution_id)
    result = response.result

    print("=" * 70)
    print(f"[{invoice_path.name}] Workflow completed!")

    if isinstance(result, dict):
        decision = result.get("decision", "unknown")
        total_amount = result.get("total_amount", 0)
        required_approval = result.get("required_human_approval", False)

        print(f"[{invoice_path.name}] DECISION: {decision}")
        print(f"[{invoice_path.name}] Total Amount: {total_amount} EUR")
        print(f"[{invoice_path.name}] Required Human Approval: {required_approval}")
        print("-" * 70)

        print(f"[{invoice_path.name}] EXTRACTED DATA:")
        extracted = result.get("extracted_data", result)
        print(json.dumps(extracted, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    print("=" * 70)


async def main() -> None:
    """Run OCR workflows on all invoices in the local invoices folder."""
    client = WorkflowsClient(
        base_url=os.environ["SERVER_URL"],
        api_key=os.environ["MISTRAL_API_KEY"],
    )

    invoice_paths = sorted(INVOICES_DIR.glob("*.jpg"))
    print(f"Found {len(invoice_paths)} invoices in {INVOICES_DIR}")
    print(f"Launching {len(invoice_paths)} workflows in parallel...\n")

    await asyncio.gather(*(run_single(client, path) for path in invoice_paths))


if __name__ == "__main__":
    asyncio.run(main())
