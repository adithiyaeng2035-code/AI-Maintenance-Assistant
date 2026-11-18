"""Tools the LLM can call. Uses a small in-memory dataset so you can run it anywhere.
Later you can swap EQUIPMENT for a real MySQL/SQLite query without touching agent.py."""
import json

EQUIPMENT = {
    "M-101": {"name": "Hydraulic Press", "temp_c": 92, "vibration_mm_s": 7.1, "failure_probability": 0.84},
    "M-102": {"name": "Conveyor Motor", "temp_c": 61, "vibration_mm_s": 2.3, "failure_probability": 0.12},
    "M-103": {"name": "CNC Lathe", "temp_c": 78, "vibration_mm_s": 4.8, "failure_probability": 0.55},
    "M-104": {"name": "Cooling Pump", "temp_c": 88, "vibration_mm_s": 6.5, "failure_probability": 0.77},
    "M-105": {"name": "Packaging Robot", "temp_c": 55, "vibration_mm_s": 1.9, "failure_probability": 0.08},
}


def get_failure_risk(equipment_id: str) -> dict:
    item = EQUIPMENT.get(equipment_id)
    if item is None:
        return {"error": f"Unknown equipment id: {equipment_id}"}
    return {"equipment_id": equipment_id, **item}


def list_high_risk_equipment(threshold: float = 0.7) -> dict:
    high = [
        {"equipment_id": eid, "name": v["name"], "failure_probability": v["failure_probability"]}
        for eid, v in EQUIPMENT.items()
        if v["failure_probability"] >= threshold
    ]
    return {"threshold": threshold, "equipment": high}


TOOL_FUNCS = {
    "get_failure_risk": get_failure_risk,
    "list_high_risk_equipment": list_high_risk_equipment,
}

# JSON schemas the model sees
TOOLS = [
    {
        "name": "get_failure_risk",
        "description": "Get sensor readings and failure probability (0-1) for one machine by its id, e.g. M-101.",
        "input_schema": {
            "type": "object",
            "properties": {"equipment_id": {"type": "string", "description": "Machine id like M-101"}},
            "required": ["equipment_id"],
        },
    },
    {
        "name": "list_high_risk_equipment",
        "description": "List all machines whose failure probability is at or above a threshold (default 0.7).",
        "input_schema": {
            "type": "object",
            "properties": {"threshold": {"type": "number", "description": "Value between 0 and 1"}},
        },
    },
]


def run_tool(name: str, args: dict) -> str:
    """Run a tool safely and always return a JSON string (errors included),
    so one bad tool call never crashes the agent loop."""
    func = TOOL_FUNCS.get(name)
    if func is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return json.dumps(func(**args))
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Tool {name} failed: {exc}"})
