"""Trigger the streaming durable agent and try to observe real-time output.

Attempts two observation methods:
1. SSE event stream (/events/stream) for real-time token streaming
2. Falls back to simple completion polling if SSE is unavailable
"""

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
from mistralai_workflows.protocol.v1.streaming import StreamEventsQueryParams
from workflows.workflow.run import OCRWorkflowInput

load_dotenv()


async def stream_events_sse(client: WorkflowsClient, execution_id: str, done: asyncio.Event) -> bool:
    """Subscribe to SSE events and print Task progress in real time. Returns True if any events were received."""
    params = StreamEventsQueryParams(
        workflow_exec_id=execution_id,
        scope="*",
        stream="*",
    )

    received_any = False
    last_text = ""

    try:
        async for event in client.stream_events(params):
            if done.is_set():
                break

            received_any = True
            data = event.data
            if not isinstance(data, dict):
                print(f"  [event] {event.stream}: {data}")
                continue

            event_type = data.get("event_type", "")
            attributes = data.get("attributes", {})

            if "CUSTOM_TASK_IN_PROGRESS" in event_type:
                if isinstance(attributes, dict):
                    payload = attributes.get("payload", {})
                    value = payload.get("value") if isinstance(payload, dict) else None
                    new_text = ""
                    if isinstance(value, dict):
                        chunks = value.get("contentChunks", [])
                        for chunk in chunks:
                            if isinstance(chunk, dict) and "text" in chunk:
                                new_text = chunk["text"]
                    elif isinstance(value, list):
                        for patch in value:
                            if isinstance(patch, dict) and patch.get("path") == "/contentChunks/0/text":
                                new_text = patch.get("value", "")
                    if new_text and new_text != last_text:
                        delta = new_text[len(last_text):] if new_text.startswith(last_text) else new_text
                        print(delta, end="", flush=True)
                        last_text = new_text

            elif "CUSTOM_TASK_STARTED" in event_type:
                task_type = attributes.get("custom_task_type", "") if isinstance(attributes, dict) else ""
                print(f"\n  [task started: {task_type or event_type}]")
                last_text = ""

            elif "CUSTOM_TASK_COMPLETED" in event_type:
                task_type = attributes.get("custom_task_type", "") if isinstance(attributes, dict) else ""
                print(f"\n  [task completed: {task_type or event_type}]")
                last_text = ""

            elif "CUSTOM_TASK_FAILED" in event_type:
                print(f"\n  [task failed: {event_type}]")

            elif "ACTIVITY_TASK" in event_type:
                activity_name = attributes.get("activity_name", "") if isinstance(attributes, dict) else ""
                status = "started" if "STARTED" in event_type else "completed" if "COMPLETED" in event_type else event_type
                print(f"  [activity {status}: {activity_name}]")

            elif "WORKFLOW" in event_type:
                print(f"  [{event_type}]")

    except asyncio.CancelledError:
        pass
    except Exception as e:
        if received_any:
            print(f"\n  [stream ended: {e}]")
        else:
            print(f"  SSE stream not available: {type(e).__name__}: {e}")

    return received_any


async def main() -> None:
    """Run the streaming OCR durable agent workflow."""
    client = WorkflowsClient(
        base_url=os.environ["SERVER_URL"],
        api_key=os.environ["MISTRAL_API_KEY"],
    )

    invoices_dir = Path(__file__).resolve().parents[2] / "invoices"
    document_path = str(sorted(invoices_dir.glob("*.jpg"))[0])
    execution_id = uuid.uuid4().hex

    print("Starting streaming durable agent workflow...")
    print(f"Document: {document_path}")
    print(f"Execution ID: {execution_id}")

    await client.execute_workflow(
        workflow_identifier="ocr_durable_agent_streaming",
        input_data=OCRWorkflowInput(document_path=document_path),
        execution_id=execution_id,
    )

    print("Workflow started.\n")

    # Try SSE streaming concurrently with completion polling
    done = asyncio.Event()

    print("--- Attempting live stream (SSE) ---\n")
    sse_task = asyncio.create_task(stream_events_sse(client, execution_id, done))
    wait_task = asyncio.create_task(client.wait_for_workflow_completion(execution_id))

    response = await wait_task
    await asyncio.sleep(2)
    done.set()
    sse_task.cancel()
    try:
        received = await sse_task
    except asyncio.CancelledError:
        received = False

    if not received:
        print("\n  (No streaming events received — SSE endpoint may not be available)")

    print("\n--- End of live stream ---")

    result = response.result
    print("\nWorkflow completed!")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
