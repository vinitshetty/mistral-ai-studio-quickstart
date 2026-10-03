"""Send approval signal to a running OCR workflow."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from mistralai.client import Mistral
from pydantic import BaseModel

load_dotenv()


class ApprovalInput(BaseModel):
    """Input for the approval signal."""
    approved: bool


async def send_approval(execution_id: str, approved: bool = True) -> None:
    """Send approval signal to a workflow execution."""
    client = Mistral(
        server_url=os.environ["SERVER_URL"],
        api_key=os.environ["MISTRAL_API_KEY"],
    )

    await client.workflows.executions.signal_workflow_execution_async(
        execution_id=execution_id,
        name="approve",
        input=ApprovalInput(approved=approved),
    )

    status = "APPROVED" if approved else "REJECTED"
    print(f"Sent {status} signal to execution: {execution_id}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python workflows/utils/approve.py <execution_id> [--reject]")
        sys.exit(1)

    exec_id = sys.argv[1]
    approved = "--reject" not in sys.argv

    asyncio.run(send_approval(exec_id, approved))
