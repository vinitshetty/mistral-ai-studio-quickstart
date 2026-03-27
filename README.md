---
id: mistral-ai-studio-quickstart
summary: Build, deploy, and monitor your first AI agent using Mistral AI Studio — from prompt to production in one session.
tags:
  - mistral
  - ai-studio
  - agents
  - document-ai
  - workflows
  - llmops
  - python
level: Beginner
estimated_time: 45 minutes
---

# Quickstart: Build & Deploy AI Agents with Mistral AI Studio

## What You'll Build

By the end of this guide, you will have:

- A **structured extraction agent** that parses invoices from raw email text
- A **deployed agent** accessible to your entire organization via Le Chat
- A **monitoring + evaluation pipeline** using Observers and LLM-as-a-Judge
- A **Document AI workflow** that extends your agent to handle PDF invoices at scale

---

## Step 1: Setup

### Prerequisites

- A [Mistral AI account](https://console.mistral.ai/) with access to AI Studio
- Python 3.10+ installed
- `uv` package manager (`pip install uv`)
- Basic familiarity with Python and REST APIs

### What You'll Learn

- How to prompt, iterate, and constrain LLM output in the Playground
- How to deploy an agent to Le Chat for org-wide access
- How to monitor live interactions and build evaluation datasets
- How to use Document AI for unstructured PDF processing
- How to orchestrate multi-step AI pipelines using Mistral Workflows

### What You'll Build

An end-to-end invoice processing system — starting from a free-text email, ending in a scalable, observable, production-grade workflow.

<img width="1470" height="732" alt="image" src="https://github.com/user-attachments/assets/384be92d-4287-4c8b-9749-4c5f173a2eba" />


---

## Step 2: Explore the Playground

The Playground is your scratchpad for prompt engineering before you commit to an agent.

**Navigate to:** AI Studio » **Playground**

Paste this sample vendor email into the prompt input:

```
Hey there! Hope you're doing well. Just sending over the bill for last week's
catering from Downtown Delights. It came out to $452.10. Let me know when
you've sent the wire!
```

Send the prompt:

```
Help me process this invoice email.
```

> **Note:** The raw response will be conversational and inconsistent — not suitable for automation or database ingestion. The next two steps fix that.

### Verification

You should see a freeform text response in the Playground output panel. This is your baseline before adding structure.

---

## Step 3: Configure Your Agent with Instructions & Structured Output

### 3a — Add System Instructions

In the Playground, open the **Instructions** panel and paste:

```
Act as a specialized data extraction assistant.

Your task is to process the following email text and extract the 'supplier_name' and the 'total_amount'.

Rules:
1. Extract 'supplier_name' as a string.
2. Extract 'total_amount' as a float (number only, remove currency symbols).
3. If information is missing, use null.
4. Output the result strictly in JSON format with the two fields 'supplier_name' and 'total_amount'. Don't add a "properties" field.

Email Text:
```

### 3b — Enforce a JSON Response Format

Open **Response Format** and use the visual builder to define this schema:

```json
{
  "type": "object",
  "required": ["supplier_name", "total_amount"],
  "properties": {
    "supplier_name": {
      "type": "string",
      "description": ""
    },
    "total_amount": {
      "type": "number",
      "description": ""
    }
  }
}
```

Re-run the same email prompt with these settings applied.

### Verification

Your output should now be a clean, predictable JSON object:

```json
{
  "supplier_name": "Downtown Delights",
  "total_amount": 452.10
}
```

---

## Step 4: Deploy Your Agent

Once you're satisfied with the output, convert your configured Playground session into a persistent agent.

1. Click **Create Agent** in the top-right of the Playground
2. Name it `invoice-extractor-v1`
3. Choose one of two deployment paths:

| Path | Use Case |
|---|---|
| **API** | Integrate into your codebase via the Mistral SDK |
| **Deploy to Le Chat** | Instant org-wide access via chat interface |

Click **Deploy to Le Chat**, then **Open in Le Chat** to test live.

Test with a second invoice:

```
Hey, this is "Boulangerie de Paris", you owe me 5€ for the croissants
```

### Verification

Le Chat returns a structured JSON response from your deployed agent. Your agent is now live and accessible across your organization.



---

## Step 5: Monitor & Evaluate Your Agent

### 5a — Observe Live Traffic

**Navigate to:** AI Studio » **Observe** » **Explorer**

You'll see every interaction your agent has processed. Identify successful extractions and flag them for your evaluation dataset.

1. Select the two successful invoice interactions
2. Click **Add to Dataset**
3. Name the dataset `my_invoices_dataset`

> **Note:** This dataset becomes your ground truth — the foundation for continuous improvement of your agent.

### 5b — Create an LLM-as-a-Judge Evaluator

1. Navigate to **Evaluate** » **Create Judge**
2. Set **Source** to `Dataset` and select `my_invoices_dataset`
3. Define your binary judge criteria:

**Correct extraction:**
```
The extraction of the fields 'total_amount' and 'supplier_name' was correct.
```

**Incorrect extraction:**
```
The extraction of the fields 'total_amount' or 'supplier_name' was not correct.
```

4. Click **Try It** to run the judge against your dataset
5. Review outputs and add labels via **Expected Output** to refine judge accuracy

### Verification

The judge runs against your dataset and produces a pass/fail score per entry. You can now close the loop: label data → fine-tune → re-evaluate.

---

## Step 6: Add Document AI for PDF Invoices

Your agent currently handles email text. Let's extend it to process PDF invoices.

Clone the workflow repository to find invoice samples:

```bash
git clone https://github.com/vinitshetty/mistral-ai-studio-quickstart.git
cd mistral-ai-studio-quickstart/invoices
```

**Navigate to:** AI Studio » **Document AI**

Upload a sample PDF invoice and observe the structured extraction output in the GUI.

> **Note:** Document AI handles OCR, layout parsing, and multi-page PDFs automatically — no preprocessing required.

### Verification

You should see structured key-value extraction from your PDF invoice, identical in format to the email-based output.

---

## Step 7: Orchestrate with Mistral Workflows

For production invoice processing — with parallelism, retries, human-in-the-loop approvals, and failure recovery — use **Mistral Workflows**.

> **What is a Workflow?** Workflows is Mistral's framework for orchestrating multi-step AI processes. Each step can be a different model, API call, or custom logic function. A central durable execution engine persists every state automatically — failures trigger retries, and long-running pipelines can pause and resume without data loss.

### 7a — Set Up the Project

Clone the example workflow repository and install dependencies:

```bash
cd mistral-ai-studio-quickstart
uv sync
```

Go to [Mistral AI Studio](https://console.mistral.ai/) » **API Keys** and generate a new API key. Then add it to the `.env.sample` file and rename it to `.env`:

```bash
cp .env.sample .env
# Open .env and set your MISTRAL_API_KEY
```

### 7b — Use Mistral Code to Understand the Repo

Open the project in VS Code with the **Mistral Code** assistant enabled. Ask it:

```
What's in this repository?
```

### 7c — Start the Workflow Server and Worker

In two separate terminal windows:

**Terminal 1 — Start the worker:**
```bash
uv run python workflows/workflow/worker.py
```

**Terminal 2 — Run the workflow:**
```bash
uv run python workflows/workflow/run.py
```

To test resume behavior (simulating a crash mid-execution):

```bash
uv run python workflows/workflow/run_w_resume.py <your-batch-id>
```

> **Warning:** Use the same `batch-id` to resume an interrupted run. Using a new ID starts a fresh execution.

### 7d — Publish to AI Studio

Once your workflow is running locally, publish it to AI Studio so users can trigger it directly from Le Chat:

**Navigate to:** AI Studio » **Workflows** » **Publish** » **Assist**

### Verification

Your workflow appears in Le Chat as a callable assistant. Users can trigger multi-step invoice processing — including PDF parsing, extraction, validation, and logging — from a single chat message.

<img width="1274" height="599" alt="image" src="https://github.com/user-attachments/assets/28da7df9-8401-400b-be25-eb321489abb5" />

<img width="1435" height="616" alt="image" src="https://github.com/user-attachments/assets/b0ff703f-13e0-4d03-96e9-130622dddb4f" />

---


## Step 8: Summary & Cleanup

### What You Built

| Component | Capability |
|---|---|
| Playground Agent | Structured JSON extraction from email text |
| Le Chat Deployment | Org-wide agent access, no code required |
| Observer + Dataset | Live monitoring and ground truth collection |
| LLM-as-a-Judge | Automated evaluation pipeline |
| Document AI | PDF invoice processing |
| Workflow | Durable, scalable, multi-step production pipeline |

### Cleanup

If you deployed resources you no longer need:

- **Delete the agent:** AI Studio » **Agents** » Select `invoice-extractor-v1` » **Delete**
- **Remove the dataset:** AI Studio » **Datasets** » `my_invoices_dataset` » **Delete**
- **Stop the workflow worker:** `Ctrl+C` in Terminal 1

> **Note:** Le Chat deployments and datasets do not incur compute costs when idle, but removing unused agents keeps your workspace clean.

---

## Related Resources

- [Mistral AI Documentation](https://docs.mistral.ai/)
- [Mistral Python SDK](https://github.com/mistralai/client-python)
- [Document AI Guide](https://docs.mistral.ai/capabilities/document/)
- [Le Chat](https://chat.mistral.ai/)
- [Mistral API Reference](https://docs.mistral.ai/api/)
