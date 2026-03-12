import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

# Add project root to sys.path for cross-folder imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from mistralai_workflows import WorkflowsClient

from workflows.workflow.run import OCRWorkflowInput

load_dotenv()


async def main() -> None:
    """Run the OCR durable agent workflow."""
    client = WorkflowsClient(
        base_url=os.environ["SERVER_URL"],
        api_key=os.environ["MISTRAL_API_KEY"],
    )

    document_url = "https://raw.githubusercontent.com/geoffroydautichamp/demo-workflows/master/invoices/batch1-1473.jpg"
    execution_id = uuid.uuid4().hex

    print("Starting durable agent workflow...")
    print(f"Document: {document_url}")
    print(f"Execution ID: {execution_id}\n")

    await client.execute_workflow(
        workflow_identifier="ocr_durable_agent",
        input_data=OCRWorkflowInput(document_url=document_url),
        execution_id=execution_id,
    )

    print("Workflow started. Waiting for completion...")
    response = await client.wait_for_workflow_completion(execution_id)
    result = response.result

    print("Workflow completed!")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())