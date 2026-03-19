"""OCR Invoice Durable Agent - Uses Mistral's Durable Agent feature to orchestrate invoice processing."""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path for cross-folder imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os

from dotenv import load_dotenv

# load_dotenv must run before importing mistralai_workflows,
# because the config is read from env vars at import time.
load_dotenv()

import mistralai  # noqa: E402
import mistralai_workflows as workflows  # noqa: E402
import mistralai_workflows.core.encoding.payload_encoder as payload_encoder  # noqa: E402
import mistralai_workflows.core.temporal.payload_codec as payload_codec  # noqa: E402
import mistralai_workflows.core.temporal.payload_converter as payload_converter  # noqa: E402
import mistralai_workflows.plugins.mistralai as workflows_mistralai  # noqa: E402
import mistralai_workflows.plugins.mistralai.activities  # noqa: E402, F401 - registers plugin activities on the worker
from pydantic import BaseModel  # noqa: E402

from workflows.workflow.worker import (  # noqa: E402
    DocumentInput,
    extract_invoice_data,
    process_document_ocr,
    THRESHOLD,
)

api_key = os.environ.get("MISTRAL_API_KEY")

# Encoding format compatibility patch
NEW_ENCODING = "json/wf_v1"
payload_encoder.CUSTOM_ENCODING_FORMAT = NEW_ENCODING
payload_codec.CUSTOM_ENCODING_FORMAT = NEW_ENCODING
payload_converter.CUSTOM_ENCODING_FORMAT = NEW_ENCODING
payload_converter.WithContextJSONPayloadConverter.encoding = property(  # type: ignore # noqa
    lambda self: NEW_ENCODING
)


class PdfDoc(BaseModel):
    document_path: str


@workflows.workflow.define(name="ocr_durable_agent")
class OCRDurableAgent:
    def __init__(self) -> None:
        self.human_approved = False
        self._approval_received = False

    @workflows.workflow.signal(name="approve", description="Human approval signal")
    async def handle_approval(self, approved: bool) -> None:
        """Signal handler for human approval."""
        self.human_approved = approved
        self._approval_received = True
        print(f"Received approval signal: {approved}")

    @workflows.workflow.entrypoint
    async def entrypoint(self, document_path: str) -> dict:
        # Step 1: Extract raw text from document via OCR
        ocr_result = await process_document_ocr(DocumentInput(document_path=document_path))

        # Step 2: Extract structured invoice data using LLM
        invoice_data = await extract_invoice_data(ocr_result)

        # Step 3: Generate a natural language summary using the durable agent
        session = workflows_mistralai.RemoteSession(raise_on_tool_fail=False)
        agent = workflows_mistralai.Agent(
            model="mistral-medium-latest",
            name="ocr-invoice-agent",
            description="Agent that summarizes extracted invoice data",
            instructions=(
                "You are an invoice assistant. "
                "Provide a concise, human-readable summary of the invoice data provided."
            ),
        )

        outputs = await workflows_mistralai.Runner.run(
            agent=agent,
            inputs=f"Summarize this invoice:\n{invoice_data.model_dump_json(indent=2)}",
            session=session,
        )

        answer = "\n".join([
            output.text for output in outputs
            if isinstance(output, mistralai.TextChunk)
        ])

        # Step 4: Human-in-the-loop validation for high-value invoices
        requires_approval = invoice_data.total_amount > THRESHOLD
        if requires_approval:
            print(
                f"Invoice amount {invoice_data.total_amount} exceeds threshold {THRESHOLD}. "
                "Waiting for human approval..."
            )
            await workflows.workflow.wait_condition(lambda: self._approval_received is True)
            decision = self.human_approved
        else:
            decision = True

        return {
            "answer": answer,
            "decision": decision,
            "total_amount": invoice_data.total_amount,
            "required_human_approval": requires_approval,
        }


async def main() -> None:
    # Run the worker — config discovery fetches Temporal settings from the Mistral API
    await workflows.run_worker(
        workflows=[OCRDurableAgent],
        api_key=api_key,
    )


if __name__ == "__main__":
    asyncio.run(main())
