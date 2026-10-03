"""OCR Invoice Workflow - Extracts structured data from PDF invoices using Mistral OCR."""

import asyncio
import base64
import os
import socket
import uuid
from typing import Any

from dotenv import load_dotenv
from temporalio import activity, workflow

# load_dotenv must run before importing mistralai.workflows,
# because the config is read from env vars at import time.
load_dotenv()

from pathlib import Path  # noqa: E402
from urllib.parse import urlparse  # noqa: E402

import aiohttp  # noqa: E402
import mistralai.workflows as workflows  # noqa: E402
import mistralai.workflows.plugins.mistralai as workflows_mistralai  # noqa: E402
import pydantic  # noqa: E402
from mistralai.client import Mistral  # noqa: E402
from mistralai.client.models import SystemMessage, UserMessage  # noqa: E402
from mistralai.workflows.plugins.mistralai.lechat import (  # noqa: E402
    FormInput,
    SingleChoice,
)

server_url = os.environ.get("SERVER_URL")
api_key = os.environ.get("MISTRAL_API_KEY")

THRESHOLD = 3000


# Data Models

class InvoiceApprovalForm(FormInput):
    """Form for human approval of an invoice."""

    decision: str = SingleChoice(  # type: ignore[assignment]
        options=[("approve", "Approve"), ("reject", "Reject")],
        description="Approve or reject this invoice",
    )


class DocumentInput(pydantic.BaseModel):
    """Input document path for OCR processing."""

    document_path: str


class OCRResponse(pydantic.BaseModel):
    """Extracted invoice data and raw text."""
    raw_text: str


class InvoiceData(pydantic.BaseModel):
    """Structured invoice data extracted by LLM."""
    invoice_number: str
    date: str
    total_amount: float
    bank_details: str
    supplier: str
    invoice_category: str

class EnrichedInvoiceData(InvoiceData):
    salesforce_id: str
    date_of_contract: str
    point_of_contact_email: str
    point_of_contact: str


class WorkflowResult(pydantic.BaseModel):
    """Final workflow result."""
    extracted_data: dict[str, Any]
    decision: bool
    total_amount: float
    required_human_approval: bool
    invoice_category: str


client = Mistral(
    server_url=server_url,
    api_key=api_key
)

# Activities


async def fetch_url(url: str) -> bytes:
    """Fetch content from a URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.read()


def is_url(path: str) -> bool:
    """Check if a path is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


@workflows.activity()
async def process_document_ocr(doc: DocumentInput) -> OCRResponse:
    """Extract structured data from a document using Mistral OCR."""

    document_path = doc.document_path

    # Get activity execution info
    info = activity.info()
    print(
        f"[Worker {socket.gethostname()}:{os.getpid()}] "
        f"Activity '{info.activity_type}' (id={info.activity_id}) "
        f"on task_queue='{info.task_queue}', attempt={info.attempt}, "
        f"workflow_id='{info.workflow_id}'"
    )

    print(f"Reading document from: {document_path}")
    
    # Handle both local files and URLs
    if is_url(document_path):
        file_bytes = await fetch_url(document_path)
    else:
        file_bytes = Path(document_path).read_bytes()
    
    base64_file = base64.b64encode(file_bytes).decode("utf-8")
    print(f"Read {len(file_bytes)} bytes")

    # Determine file type from URL or local path
    if is_url(document_path):
        # Extract file extension from URL
        parsed = urlparse(document_path)
        path_part = parsed.path
        suffix = Path(path_part).suffix.lower() if path_part else ""
    else:
        suffix = Path(document_path).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        document_payload = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_file}",
        }
    elif suffix == ".png":
        document_payload = {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{base64_file}",
        }
    else:
        document_payload = {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_file}",
        }

    print("Running OCR with structured extraction...")
    ocr_response = await client.ocr.process_async(
        model="mistral-ocr-latest",
        document=document_payload,  # type: ignore[arg-type]
        include_image_base64=False,
    )

    # Log the OCR response for debugging
    print("OCR Response:", ocr_response)

    # Extract structured data and raw text
    raw_text_parts = []

    for page in ocr_response.pages:
        raw_text_parts.append(page.markdown)

    return OCRResponse(raw_text="\n\n".join(raw_text_parts))


@workflows.activity()
async def extract_invoice_data(ocr_result: OCRResponse) -> InvoiceData:
    """Extract structured invoice data from raw text using LLM."""

    # Get activity execution info
    info = activity.info()
    print(
        f"[Worker {socket.gethostname()}:{os.getpid()}] "
        f"Activity '{info.activity_type}' (id={info.activity_id}) "
        f"on task_queue='{info.task_queue}', attempt={info.attempt}, "
        f"workflow_id='{info.workflow_id}'"
    )

    SYSTEM_PROMPT = """
    You are an agent specialised in extracting invoice data and categorising invoices. You will be provided with invoices under markdown format, and will be tasked to extract relevant information and categorise the invoice.

    Required fields:
    - invoice_number: The invoice number
    - date: The invoice date (format: YYYY-MM-DD)
    - total_amount: The total amount of the invoice (numeric value)
    - bank_details: The bank account details (RIB)
    - supplier: The supplier who sent the bill
    - invoice_category: The category of the invoice. One of: financial_expense (bank fees, etc.), subscriptions (phones, software licences, etc), furniture (office supplies), raw_materials (purchase of good for production), rent (office rentals), professional_services (accountants, lawyers, consultants), expenses (travel, meals), marketing (ads, content creation), or unknown

    If any field cannot be determined, use "unknown" for strings or 0 for numeric values.
    """

    USER_PROMPT = """
    Extract the following information from the invoice text below:

    Invoice text:
    {raw_text}

    ###
    Provide the extracted information in the required format.
    """
    SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "date",
            "invoice_number",
            "total_amount",
            "supplier",
            "invoice_category",
            "bank_details",
        ],
        "properties": {
            "date": {"type": "string", "description": "Date of the invoice"},
            "supplier": {"type": "string", "description": "supplier"},
            "bank_details": {
                "type": "string",
                "description": "The bank details of the supplier",
            },
            "total_amount": {"type": "number", "description": ""},
            "invoice_number": {"type": "string", "description": ""},
            "invoice_category": {
                "type": "string",
                "enum": [
                    "financial_expense",
                    "subscriptions",
                    "furniture",
                    "raw_materials",
                    "rent",
                    "professional_services",
                    "expenses",
                    "marketing",
                    "unknown",
                ],
                "description": "Category of the spend",
            },
        },
    }

    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[  # type: ignore # noqa
                SystemMessage(content=SYSTEM_PROMPT),
                UserMessage(content=USER_PROMPT.format(raw_text=ocr_result)),
            ],
            response_format={  # type: ignore # noqa
                "type": "json_schema",
                "json_schema": {  # type: ignore # noqa
                    "name": "invoice_data",
                    "schema": SCHEMA,
                    "strict": True,
                },
            },
        )

        raw_content = response.choices[0].message.content
        if not isinstance(raw_content, str):
            raise ValueError(f"Unexpected content type: {type(raw_content)}")

        invoice_data = InvoiceData.model_validate_json(raw_content)
    except Exception as e:
        print(f"Error extracting invoice data: {e}")
        # Return default values if extraction fails
        return InvoiceData(
            invoice_number="unknown",
            date="unknown",
            total_amount=10.0,
            bank_details="unknown",
            supplier="unknown",
            invoice_category="unknown",
        )
    print("Extracted invoice data:", invoice_data)
    return invoice_data


@workflows.activity()
async def data_enrichment_with_mcp(invoice_data: InvoiceData) -> EnrichedInvoiceData:
    """Make a call to a MCP server with the extracted invoice data."""
    # Get activity execution info
    info = activity.info()
    print(
        f"Activity '{info.activity_type}' (id={info.activity_id}) "
        f"on task_queue='{info.task_queue}', attempt={info.attempt}, "
        f"workflow_id='{info.workflow_id}')"
    )

    # Simulate a call to a MCP server
    return EnrichedInvoiceData(
        **dict(invoice_data),
        salesforce_id=str(uuid.uuid1()),
        date_of_contract="2023-01-01",
        point_of_contact_email="john.doe@example.com",
        point_of_contact="John Doe",
    )


@workflows.activity()
async def send_to_validation(invoice_data: InvoiceData) -> bool:
    """Send invoice data to validation system."""
    # Get activity execution info
    info = activity.info()
    print(
        f"Activity '{info.activity_type}' (id={info.activity_id}) "
        f"on task_queue='{info.task_queue}', attempt={info.attempt}, "
        f"workflow_id='{info.workflow_id}')"
    )
    # import time

    # time.sleep(1)
    await asyncio.sleep(1)
    # Simulate a call to a MCP server
    return True


# Workflow Definition
@workflows.workflow.define(
    name="ocr_invoice_workflow_test",
    workflow_display_name="OCR Invoice Workflow Test",
    workflow_description="Extract structured data from PDF invoices with human approval for high amounts",
)
class OCRDocumentWorkflow(workflows.InteractiveWorkflow):
    """Extracts invoice data from PDFs. Requires human approval for amounts >= threshold."""

    @workflows.workflow.signal(name="approve", description="Approve or reject the invoice (AIStudio)")
    async def handle_approval(self, approved: bool) -> None:
        """AIStudio signal path: bridges into the pending wait_for_input task."""
        decision = "approve" if approved else "reject"
        for pending in self._pending_inputs.values():  # type: ignore[attr-defined]
            pending.input = {"decision": decision}
            pending.has_received_input = True
            break

    @workflows.workflow.entrypoint
    async def run(
        self, document_path: str
    ) -> workflows_mistralai.ChatAssistantWorkflowOutput:
        """Process document, extract data, and get human approval if needed."""

        info = workflow.info()
        workflow.logger.info(
            f"Workflow '{info.workflow_type}' started | "
            f"workflow_id='{info.workflow_id}', run_id='{info.run_id}', "
            f"task_queue='{info.task_queue}', attempt={info.attempt}"
        )

        # Define todo list steps
        ocr_item = workflows_mistralai.TodoListItem(
            title="OCR Extraction", description="Extract text from document"
        )
        extract_item = workflows_mistralai.TodoListItem(
            title="Data Extraction", description="Extract structured invoice data"
        )
        enrich_item = workflows_mistralai.TodoListItem(
            title="Data Enrichment", description="Enrich with CRM data"
        )
        process_item = workflows_mistralai.TodoListItem(
            title="Approval & Processing", description="Get approval and process"
        )

        async with workflows_mistralai.TodoList(
            items=[ocr_item, extract_item, enrich_item, process_item]
        ):
            # Step 1: Extract raw text from document
            async with ocr_item:
                ocr_result = await process_document_ocr(
                    DocumentInput(document_path=document_path)
                )

            # Step 2: Extract structured invoice data using LLM
            async with extract_item:
                invoice_data = await extract_invoice_data(ocr_result)

            # Step 3: Enrich with CRM data
            async with enrich_item:
                data_enriched_by_mcp = await data_enrichment_with_mcp(invoice_data)

            # Step 4: Approval & processing
            requires_approval = data_enriched_by_mcp.total_amount > THRESHOLD
            await process_item.set_status("in_progress")

            if requires_approval:
                await workflows_mistralai.send_assistant_message(
                    f"Invoice **#{invoice_data.invoice_number}** from **{invoice_data.supplier}** "
                    f"requires your approval (amount: **${invoice_data.total_amount:.2f}**)."
                )
                confirmation = await self.wait_for_input(InvoiceApprovalForm)
                decision = confirmation.decision == "approve"
            else:
                decision = True  # Auto-approve if amount <= THRESHOLD

            await process_item.set_status("done")

        status = "APPROVED" if decision else "REJECTED"
        summary = (
            f"Invoice processing complete.\n\n"
            f"**Invoice #{invoice_data.invoice_number}**\n"
            f"- Supplier: {invoice_data.supplier}\n"
            f"- Date: {invoice_data.date}\n"
            f"- Amount: ${invoice_data.total_amount:.2f}\n"
            f"- Category: {invoice_data.invoice_category}\n"
            f"- Decision: {status}\n"
            f"- Required approval: {requires_approval}"
        )
        await workflows_mistralai.send_assistant_message(summary)

        return workflows_mistralai.ChatAssistantWorkflowOutput(
            content=[workflows_mistralai.TextOutput(text=summary)]
        )

# Version STANDARD
async def main() -> None:
    """Start the OCR workflow worker."""
    print("Starting OCR Invoice Workflow worker...")

    # Run the worker — config discovery fetches Temporal settings from the Mistral API
    await workflows.run_worker(
        workflows=[OCRDocumentWorkflow],
        api_key=api_key,
    )

if __name__ == "__main__":
    asyncio.run(main())
