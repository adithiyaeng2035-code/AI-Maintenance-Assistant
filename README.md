# AI Maintenance Assistant

A small FastAPI service where an LLM answers equipment-maintenance questions by **calling tools**, checking the results, and returning **validated JSON**. The agent loop is written in plain Python (no LangChain/LangGraph), so every step is visible.

## How it works

1. `POST /ask` receives a question, e.g. "Which machines are at highest risk?"
2. The agent sends the question and the tool definitions to the model.
3. If the model asks for a tool (`get_failure_risk`, `list_high_risk_equipment`), the tool runs and the result is sent back. This repeats for at most **5 steps**.
4. The final reply is parsed and validated with **Pydantic**. If the JSON is invalid, the model is asked once more to correct it.
5. If no valid answer is produced within the step limit, the API returns **502**.

```
question -> model -> (tool call -> tool result -> model)* -> JSON -> Pydantic check -> response
```

## Project structure

| File | Purpose |
|---|---|
| `main.py` | FastAPI app (`/health`, `/ask`) |
| `agent.py` | Agent loop, step limit, JSON validation, retry on bad JSON |
| `tools.py` | Tool functions and their JSON schemas (in-memory sample data) |
| `test_agent.py` | pytest tests using a fake LLM client (no API key needed) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env          # then put your OPENAI_API_KEY in .env
```

## Run

```bash
pytest                         # 5 tests, LLM is mocked
uvicorn main:app --reload      # then open http://127.0.0.1:8000/docs
```

## Example

Request:
```json
{"question": "Which machines are at highest risk?"}
```

Response (shape; exact wording varies by model):
```json
{
  "summary": "M-101 and M-104 are high risk.",
  "machines": [
    {"machine_id": "M-101", "risk_level": "high", "failure_probability": 0.84,
     "reason": "92C temperature and 7.1 mm/s vibration"}
  ],
  "recommended_action": "Inspect M-101 and M-104 today."
}
```

<!-- TODO: add a screenshot of /docs showing a real response -->

## Design decisions

- **No agent framework:** writing the loop by hand makes the tool-calling flow explicit.
- **Step limit (5):** prevents a model from looping forever and running up cost.
- **Validate, don't trust:** model output is checked with Pydantic before it is returned.
- **Tools never crash the loop:** errors are returned to the model as JSON so it can recover.
- **Secrets:** the API key lives in `.env`, which is git-ignored.
- **Tests mock the LLM:** fast, free, and deterministic.

## A bug I hit

<!-- TODO: write 2-3 sentences about one real problem you ran into while building this
     and how you fixed it. Only write what actually happened. -->

## Possible next steps

- Replace the in-memory data with MySQL/PostgreSQL
- Add a `create_maintenance_ticket` tool
- Add a Dockerfile
- Connect to a real ML model for failure prediction
