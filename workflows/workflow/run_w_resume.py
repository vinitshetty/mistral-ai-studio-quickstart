"""Execute the OCR Invoice Workflow with resume/retry support.

Re-uses deterministic execution IDs based on invoice filename so that:
- Already completed invoices are skipped
- Running executions are attached to (not re-started)
- Failed/timed-out/cancelled executions are re-executed with the same ID
"""

import asyncio
import json
import os
from http import HTTPStatus
from pathlib import Path

from dotenv import load_dotenv
from mistralai_workflows import WorkflowsClient
from mistralai_workflows.exceptions import WorkflowsException
from mistralai_workflows.protocol.v1.workflow import WorkflowExecutionStatus
from pydantic import BaseModel

load_dotenv()

INVOICES_DIR = Path(__file__).resolve().parents[2] / "invoices"

# Statuses that mean "still in progress — attach and wait"
_RUNNING_STATUSES = {
    WorkflowExecutionStatus.RUNNING,
    WorkflowExecutionStatus.RETRYING_AFTER_ERROR,
    WorkflowExecutionStatus.CONTINUED_AS_NEW,
}

# Terminal failure statuses — safe to re-execute with the same ID
_RETRIABLE_STATUSES = {
    WorkflowExecutionStatus.FAILED,
    WorkflowExecutionStatus.TIMED_OUT,
    WorkflowExecutionStatus.CANCELED,
    WorkflowExecutionStatus.TERMINATED,
}


class OCRWorkflowInput(BaseModel):
    """Input for the OCR workflow."""
    document_path: str


async def run_single(client: WorkflowsClient, invoice_path: Path) -> None:
    """Run the OCR workflow on a single invoice, resuming if already in progress."""
    execution_id = f"invoice-{invoice_path.stem}"

    # Check existing state on the server (Temporal is the source of truth)
    should_start = False
    try:
        existing = await client.get_workflow_execution(execution_id)

        if existing.status == WorkflowExecutionStatus.COMPLETED:
            print(f"[{invoice_path.name}] Already completed, skipping.")
            return
        elif existing.status in _RUNNING_STATUSES:
            print(f"[{invoice_path.name}] Already running (status={existing.status}), attaching...")
        elif existing.status in _RETRIABLE_STATUSES:
            print(f"[{invoice_path.name}] Previous run ended with status={existing.status}, re-executing...")
            should_start = True
        else:
            print(f"[{invoice_path.name}] Unknown status={existing.status}, re-executing...")
            should_start = True

    except WorkflowsException as e:
        if e.status == HTTPStatus.NOT_FOUND:
            print(f"[{invoice_path.name}] No existing execution found, starting fresh.")
            should_start = True
        else:
            raise

    if should_start:
        print(f"[{invoice_path.name}] Execution ID: {execution_id}")
        try:
            await client.execute_workflow(
                workflow_identifier="ocr_invoice_workflow_test",
                input_data=OCRWorkflowInput(document_path=str(invoice_path)),
                execution_id=execution_id,
            )
        except WorkflowsException as e:
            if e.status == HTTPStatus.CONFLICT:
                # Race condition: another process started it between our check and execute
                print(f"[{invoice_path.name}] Conflict on start, attaching to running execution.")
            else:
                raise

    print(f"[{invoice_path.name}] Waiting for completion...")
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
    """Run OCR workflows on all invoices, skipping completed and resuming in-progress ones."""
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
