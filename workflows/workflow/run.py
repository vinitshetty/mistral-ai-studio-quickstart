"""Execute the OCR Invoice Workflow."""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from mistralai_workflows import WorkflowsClient
from pydantic import BaseModel

load_dotenv()

INVOICES_DIR = Path(__file__).resolve().parents[2] / "invoices"


class OCRWorkflowInput(BaseModel):
    """Input for the OCR workflow."""
    document_path: str


def is_url(path: str) -> bool:
    """Check if a path is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


async def run_single(client: WorkflowsClient, document_path: str) -> None:
    """Run the OCR workflow on a single document."""
    execution_id = uuid.uuid4().hex
    doc_name = os.path.basename(document_path) if not is_url(document_path) else document_path

    print(f"[{doc_name}] Starting workflow for {document_path}")
    print(f"[{doc_name}] Execution ID: {execution_id}")

    await client.execute_workflow(
        workflow_identifier="ocr_invoice_workflow_test",
        input_data=OCRWorkflowInput(document_path=document_path),
        execution_id=execution_id,
    )

    print(f"[{doc_name}] Workflow started. Waiting for completion...")
    print(f"[{doc_name}] To approve: uv run python workflows/utils/approve.py {execution_id}")

    response = await client.wait_for_workflow_completion(execution_id)
    result = response.result

    print("=" * 70)
    print(f"[{doc_name}] Workflow completed!")

    if isinstance(result, dict):
        decision = result.get("decision", "unknown")
        total_amount = result.get("total_amount", 0)
        required_approval = result.get("required_human_approval", False)

        print(f"[{doc_name}] DECISION: {decision}")
        print(f"[{doc_name}] Total Amount: {total_amount} EUR")
        print(f"[{doc_name}] Required Human Approval: {required_approval}")
        print("-" * 70)

        print(f"[{doc_name}] EXTRACTED DATA:")
        extracted = result.get("extracted_data", result)
        print(json.dumps(extracted, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    print("=" * 70)


async def main() -> None:
    """Run OCR workflows on local invoices or URL inputs."""
    client = WorkflowsClient(
        base_url=os.environ["SERVER_URL"],
        api_key=os.environ["MISTRAL_API_KEY"],
    )

    # Accept both local files and URLs from command line arguments
    if len(sys.argv) > 1:
        document_paths = sys.argv[1:]
    else:
        # Fallback to local invoices directory
        invoice_paths = sorted(INVOICES_DIR.glob("*.jpg"))
        document_paths = [str(p) for p in invoice_paths]
        print(f"Found {len(invoice_paths)} invoices in {INVOICES_DIR}")

    print(f"Launching {len(document_paths)} workflows in parallel...\n")

    await asyncio.gather(*(run_single(client, path) for path in document_paths))


if __name__ == "__main__":
    asyncio.run(main())
