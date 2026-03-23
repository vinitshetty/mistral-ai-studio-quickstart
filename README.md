# OCR Invoice Workflow Demo

Demonstrates durable workflows for invoice processing using [Mistral AI Workflows](https://docs.mistral.ai/capabilities/workflows/) (backed by Temporal).

The project showcases three approaches:
- **Standard workflow** — manually orchestrated activities with human-in-the-loop approval
- **Durable agent** — an LLM agent with tool use, orchestrated via Mistral's Agents SDK with fine-grained durability
- **Durable agent with streaming** — same as above, with real-time token-by-token observability via SSE events

## Project Structure

```
workflows/
├── workflow/                        # Standard workflow (explicit control flow)
│   ├── worker.py                    # Activities (OCR, extraction) + workflow definition
│   ├── run.py                       # Trigger workflows (batch, via Temporal)
│   └── run_local.py                 # Run locally (no Temporal)
│
├── durable_agent/                   # Durable agent (LLM-driven control flow)
│   ├── durable_agent.py             # Agent worker with RemoteSession
│   └── run_agent.py                 # Trigger the agent workflow
│
├── durable_agent_streaming/         # Durable agent with streaming observability
│   ├── durable_agent_streaming.py   # Agent worker with RemoteSession(stream=True)
│   └── run_agent_streaming.py       # Trigger + live SSE event stream
│
├── utils/                           # Shared utilities
│   ├── approve.py                   # Send approval/rejection signal
│   └── test.py                      # Quick test: OCR + extraction directly
│
└── utils.py                         # Terminate a running workflow execution
```

## Prerequisites

- Python 3.12
- A Mistral API key
- Access to Mistral's Temporal-backed workflow infrastructure

## Setup

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your MISTRAL_API_KEY and SERVER_URL
```

---

## Standard Workflow (`workflow/`)

A step-by-step workflow with explicit control flow. You define each activity and the workflow orchestrates them in sequence.

### Activities (`worker.py`)

Two activities decorated with `@workflows.activity()`:

- **`process_document_ocr`** — Downloads the document and extracts text via Mistral OCR (`mistral-ocr-latest`)
- **`extract_invoice_data`** — Sends the raw text to a Mistral agent to extract structured fields: invoice number, date, total amount, supplier, bank details, category

### Workflow (`OCRDocumentWorkflow` in `worker.py`)

1. Calls `process_document_ocr` to get raw text
2. Calls `extract_invoice_data` to get structured data
3. If `total_amount > 100 EUR`, the workflow **pauses** and waits for a human approval signal via Temporal
4. Returns the extracted data + approval decision

### Scripts

**Start the worker:**
```bash
uv run python workflows/workflow/worker.py
```

**Trigger a batch of workflows** (invoices 1472–1490 in parallel):
```bash
uv run python workflows/workflow/run.py

uv run python workflows/workflow/run_w_resume.py <run-id>
```

**Run locally without Temporal** (useful for development):
```bash
uv run python workflows/workflow/run_local.py
```

**Approve or reject a paused workflow:**
```bash
uv run python workflows/utils/approve.py <execution_id>
uv run python workflows/utils/approve.py <execution_id> --reject
```

---

## Durable Agent (`durable_agent/`)

Instead of coding each step explicitly, an LLM agent autonomously decides which tools to call. Uses `RemoteSession` so each LLM call and tool invocation is a separate Temporal activity.

### How it works (`durable_agent.py`)

1. Creates a `RemoteSession(raise_on_tool_fail=False)` — conversation state lives on Mistral's servers, and malformed tool calls are returned as errors to the agent (letting it retry) instead of failing the workflow
2. Configures an agent (`mistral-medium-latest`) with instructions and two tools: `process_document_ocr` and `extract_invoice_data`
3. Runs the agent via `Runner.run()` — the agent loop is decomposed into individual Temporal activities
4. If the worker crashes mid-execution, Temporal replays completed activities from history and resumes from the last checkpoint

### Scripts

**Start the worker:**
```bash
uv run python workflows/durable_agent/durable_agent.py
```

**Trigger the agent workflow** (single invoice):
```bash
uv run python workflows/durable_agent/run_agent.py
```

---

## Durable Agent with Streaming (`durable_agent_streaming/`)

Same architecture as the durable agent, but with `RemoteSession(stream=True)` to enable real-time observability of the LLM's output.

### How streaming works

When `stream=True`, the session uses Mistral's streaming conversation endpoints. As the LLM generates tokens:

1. The streaming activity receives tokens incrementally from the Mistral API
2. Inside the activity, a `Task` context manager publishes `CustomTaskInProgress` events to the Workflows API
3. Each event contains a JSON patch updating the `contentChunks` text with the latest accumulated content
4. External consumers (UIs, CLIs, dashboards) subscribe to these events via SSE and render the output as it's generated

The activity still returns the complete response at the end — streaming adds observability without changing the final result.

### `run_agent_streaming.py` — Trigger with live output

The trigger script subscribes to the SSE event stream (`/events/stream`) concurrently with polling for workflow completion:

- **`WORKFLOW_EXECUTION_STARTED`** — Workflow has begun
- **`ACTIVITY_TASK_STARTED` / `ACTIVITY_TASK_COMPLETED`** — Activity lifecycle events (agent creation, conversation calls, tool execution)
- **`CUSTOM_TASK_STARTED`** — An LLM response is beginning
- **`CUSTOM_TASK_IN_PROGRESS`** — Extracts the text delta from JSON patches on `contentChunks` and prints new tokens as they arrive
- **`CUSTOM_TASK_COMPLETED`** — The LLM response is complete

### Scripts

**Start the worker:**
```bash
uv run python workflows/durable_agent_streaming/durable_agent_streaming.py
```

**Trigger with live streaming output:**
```bash
uv run python workflows/durable_agent_streaming/run_agent_streaming.py
```

---

## Utilities

### `utils/approve.py` — Human Approval Signal

Sends an approval or rejection signal to a running standard workflow when an invoice exceeds the approval threshold.

```bash
uv run python workflows/utils/approve.py <execution_id>
uv run python workflows/utils/approve.py <execution_id> --reject
```

### `utils/test.py` — Quick OCR Test

Runs OCR extraction and invoice data parsing directly — no workflow, no Temporal. Useful for verifying activities work independently.

```bash
uv run python workflows/utils/test.py
```

### `utils.py` — Terminate Execution

Terminates a running workflow execution by ID.

```bash
uv run python workflows/utils.py <execution_id>
```

---

## Comparison

| | Standard Workflow | Durable Agent | Durable Agent (Streaming) |
|---|---|---|---|
| **Control flow** | Explicit (you write the steps) | LLM-driven (agent decides) | LLM-driven |
| **Session type** | N/A (manual activities) | `RemoteSession` | `RemoteSession(stream=True)` |
| **Durability granularity** | Per activity | Per LLM call / tool call | Per LLM call / tool call |
| **Real-time observability** | No | No | Yes (SSE token streaming) |
| **Human approval** | Built-in via Temporal signals | No | No |
| **Crash recovery** | Resumes after last completed activity | Resumes after last LLM call or tool call | Resumes after last LLM call or tool call |
