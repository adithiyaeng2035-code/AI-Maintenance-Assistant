"""The agent loop, written in plain Python (no framework) so you can explain every step:
ask model -> if it wants a tool, run it -> feed the result back -> repeat -> validate final JSON."""
import json
import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from tools import TOOLS, run_tool

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_STEPS = 5

# OpenAI wants tools wrapped as {"type": "function", "function": {...}}
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOLS
]

SYSTEM_PROMPT = """You are a predictive-maintenance assistant for a factory.
Use the tools to look up real data. Never guess machine data.
When you have enough information, reply with ONLY a JSON object (no markdown, no extra text) shaped like:
{
  "summary": "one or two sentences",
  "machines": [
    {"machine_id": "M-101", "risk_level": "low|medium|high",
     "failure_probability": 0.0, "reason": "why"}
  ],
  "recommended_action": "what the maintenance team should do"
}
Use risk_level high for probability >= 0.7, medium for 0.4-0.69, low below 0.4."""


class MachineRisk(BaseModel):
    machine_id: str
    risk_level: Literal["low", "medium", "high"]
    failure_probability: float
    reason: str


class MaintenanceAnswer(BaseModel):
    summary: str
    machines: list[MachineRisk]
    recommended_action: str


class AgentError(Exception):
    """Raised when the agent cannot produce a valid answer."""


def _parse_answer(text: str) -> MaintenanceAnswer:
    cleaned = (text or "").strip()
    # Models sometimes wrap JSON in ```json fences even when told not to
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return MaintenanceAnswer.model_validate(json.loads(cleaned))


def run_agent(question: str, client=None, max_steps: int = MAX_STEPS) -> MaintenanceAnswer:
    client = client or OpenAI()  # reads OPENAI_API_KEY from the environment
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=OPENAI_TOOLS,
        )
        message = response.choices[0].message

        if message.tool_calls:
            # Record the assistant's tool request, then answer each call
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            })
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": run_tool(tc.function.name, args),
                })
            continue

        # Final answer: validate it instead of trusting it
        try:
            return _parse_answer(message.content)
        except (json.JSONDecodeError, ValidationError) as exc:
            messages.append({"role": "assistant", "content": message.content or ""})
            messages.append({
                "role": "user",
                "content": f"Your reply was not valid JSON for the required schema ({exc}). "
                           "Reply again with ONLY the corrected JSON object.",
            })

    raise AgentError(f"No valid answer after {max_steps} steps")
