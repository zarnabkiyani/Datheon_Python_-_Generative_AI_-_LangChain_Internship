# Task 1: Structured LLM Prompt Pipeline & JSON Extractor

This project reads messy, unstructured text (like an invoice or a resume) and turns it into clean, checked JSON using an LLM (Google Gemini or OpenAI).

**Notebook:** `Task1_Structured_LLM_JSON_Extractor_v2 (1).ipynb`
**Colab link:** [PASTE YOUR COLAB LINK HERE]

---

## What It Does

1. Takes a plain text document (invoice or resume).
2. Builds a prompt from a reusable template with a few-shot example.
3. Sends the prompt to the LLM (Gemini by default, OpenAI also supported) in JSON mode.
4. Checks the answer with strict Pydantic schemas.
5. If the answer is wrong, it sends the error back to the LLM and tries again (up to 3 times).
6. Saves the final valid result as a `.json` file.

---

## Tech Used

- Python 3
- `google-genai` (Google Gemini) or `openai`
- `pydantic` (data validation)
- Google Colab

---

## Features

- **Parameterized prompt templates:** one template works for different document types.
- **Few-shot prompting:** each document type has a worked example to guide the model.
- **Strict validation:**
  - Dates must be in `YYYY-MM-DD` format.
  - Currency must be a 3-letter code (like `USD`, `INR`).
  - Email must be valid.
  - Subtotal + tax must equal total.
  - An invoice must have at least one line item.
- **Auto-retry:** invalid output is sent back to the model with the error message to fix.
- **Two providers:** switch between Gemini and OpenAI by changing one line.
- **Safe API keys:** keys are read from Colab Secrets, never written in the code.

---

## How To Run (Google Colab)

1. Open the notebook in [Google Colab](https://colab.research.google.com).
2. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
3. In Colab, click the key icon (**Secrets**) on the left sidebar.
4. Add a new secret named `GOOGLE_API_KEY` and paste your key as the value.
5. Turn ON **Notebook access**.
6. Click **Runtime → Run all**.
7. When the last cell asks for input, just press **Enter** to skip it.

**Using OpenAI instead?** Add a secret named `OPENAI_API_KEY` and change this line in the config cell:

```python
PROVIDER = "openai"
```

**Got a 404 model error?** Change the model name in the config cell to the one shown in the error message:

```python
GEMINI_MODEL = "your-model-name"
```

---

## Project Structure

| Part | What it does |
|------|--------------|
| Pydantic schemas | Define and validate the JSON for `Invoice` and `Resume` |
| Prompt templates | Reusable prompt with schema, rules and a few-shot example |
| LLM connector | Calls Gemini or OpenAI in JSON mode |
| Extraction pipeline | Builds prompt, calls LLM, validates, retries if needed |
| Sample documents | One invoice and one resume to test the pipeline |
| Validation test | Shows that bad JSON is correctly rejected |

---

## Example

**Input (unstructured text):**

```
Tax Invoice No. INV/2024/0457
Issued 12 Jan 2024, payment due within 30 days (by 11 Feb 2024).
Seller: GreenLeaf Stationers Pvt Ltd, Mumbai
Buyer: Rohan Enterprises
Items:
 - A4 Paper Ream, qty 10, Rs 250 each, Rs 2,500
 - Ballpoint Pens (box), qty 5, Rs 120 each, Rs 600
Subtotal Rs 3,100 | GST 18%: Rs 558 | Grand total: Rs 3,658
All amounts in INR.
```

**Output (validated JSON):**

```json
{
  "invoice_number": "INV/2024/0457",
  "invoice_date": "2024-01-12",
  "due_date": "2024-02-11",
  "vendor_name": "GreenLeaf Stationers Pvt Ltd",
  "customer_name": "Rohan Enterprises",
  "currency": "INR",
  "line_items": [
    {
      "description": "A4 Paper Ream",
      "quantity": 10.0,
      "unit_price": 250.0,
      "amount": 2500.0
    },
    {
      "description": "Ballpoint Pens (box)",
      "quantity": 5.0,
      "unit_price": 120.0,
      "amount": 600.0
    }
  ],
  "subtotal": 3100.0,
  "tax": 558.0,
  "total": 3658.0
}
```

---

## Add Your Own Document Type

1. Create a new Pydantic class with the fields you need.
2. Add an entry to the `DOC_TYPES` dictionary with instructions and a few-shot example.
3. Call `extract("your_type", your_text)`.

---

## Deliverable

An executable notebook that parses complex unstructured text documents into validated JSON schemas.

---

## Author

zarnabkiyani
