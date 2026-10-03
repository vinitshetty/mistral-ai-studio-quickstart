"""Utility functions for managing workflow runs."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()


async def terminate_run(execution_id: str) -> None:
    """Terminate a workflow execution by ID."""
    client = Mistral(
        server_url=os.environ["SERVER_URL"],
        api_key=os.environ["MISTRAL_API_KEY"],
    )
    await client.workflows.executions.terminate_workflow_execution_async(
        execution_id=execution_id
    )
    print(f"Terminated: {execution_id}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python workflows/utils.py <execution_id>")
        sys.exit(1)
    asyncio.run(terminate_run(sys.argv[1]))
