"""Execute the OCR Invoice Workflow."""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from mistralai.client import Mistral
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


async def run_single(client: Mistral, document_path: str, deployment_name: str | None) -> None:
    """Run the OCR workflow on a single document."""
    execution_id = uuid.uuid4().hex
    doc_name = os.path.basename(document_path) if not is_url(document_path) else document_path

    print(f"[{doc_name}] Starting workflow for {document_path}")
    print(f"[{doc_name}] Execution ID: {execution_id}")

    await client.workflows.execute_workflow_async(
        workflow_identifier="ocr_invoice_workflow_test",
        input=OCRWorkflowInput(document_path=document_path),
        execution_id=execution_id,
        **({"deployment_name": deployment_name} if deployment_name else {}),
    )

    print(f"[{doc_name}] Workflow started. Waiting for completion...")
    print(f"[{doc_name}] To approve: uv run python workflows/utils/approve.py {execution_id}")

    response = await client.workflows.wait_for_workflow_completion_async(execution_id)
    result = response.result

    print("=" * 70)
    print(f"[{doc_name}] Workflow completed!")

    if isinstance(result, dict) and "content" in result:
        # ChatAssistantWorkflowOutput shape returned by the workflow
        text = "\n".join(
            item.get("text", "") for item in result["content"] if isinstance(item, dict)
        )
        print(text or json.dumps(result, indent=2, ensure_ascii=False))
    elif isinstance(result, dict):
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
    client = Mistral(
        server_url=os.environ["SERVER_URL"],
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

    deployment_name = os.environ.get("DEPLOYMENT_NAME")
    await asyncio.gather(
        *(run_single(client, path, deployment_name) for path in document_paths)
    )


if __name__ == "__main__":
    asyncio.run(main())
