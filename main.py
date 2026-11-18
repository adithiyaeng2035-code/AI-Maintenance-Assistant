from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import AgentError, MaintenanceAnswer, run_agent

app = FastAPI(title="AI Maintenance Assistant")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=MaintenanceAnswer)
def ask(req: AskRequest):
    try:
        return run_agent(req.question)
    except AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
