yes"""Tests run with a fake LLM client, so they need no API key and no network."""
import json
from types import SimpleNamespace

import pytest

from agent import AgentError, run_agent
from tools import get_failure_risk, list_high_risk_equipment, run_tool

GOOD_JSON = json.dumps({
    "summary": "M-101 is at high risk.",
    "machines": [{"machine_id": "M-101", "risk_level": "high",
                  "failure_probability": 0.84, "reason": "High temperature and vibration."}],
    "recommended_action": "Inspect M-101 today.",
})


def _wrap(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def text_response(text):
    return _wrap(SimpleNamespace(content=text, tool_calls=None))


def tool_response(name, args, tool_id="t1"):
    call = SimpleNamespace(id=tool_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return _wrap(SimpleNamespace(content=None, tool_calls=[call]))


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)  # so client.chat.completions.create(...) works

    def create(self, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


def test_tools_return_expected_data():
    assert get_failure_risk("M-101")["failure_probability"] == 0.84
    assert "error" in get_failure_risk("NOPE")
    ids = [e["equipment_id"] for e in list_high_risk_equipment(0.7)["equipment"]]
    assert "M-101" in ids and "M-102" not in ids


def test_unknown_tool_does_not_crash():
    assert "error" in json.loads(run_tool("does_not_exist", {}))


def test_agent_calls_tool_then_answers():
    client = FakeClient([tool_response("get_failure_risk", {"equipment_id": "M-101"}),
                         text_response(GOOD_JSON)])
    answer = run_agent("How risky is M-101?", client=client)
    assert answer.machines[0].risk_level == "high"
    assert client.calls == 2


def test_agent_recovers_from_bad_json():
    client = FakeClient([text_response("not json at all"), text_response(GOOD_JSON)])
    answer = run_agent("Status?", client=client)
    assert answer.machines[0].machine_id == "M-101"
    assert client.calls == 2


def test_agent_stops_at_step_limit():
    client = FakeClient([tool_response("list_high_risk_equipment", {}, f"t{i}") for i in range(10)])
    with pytest.raises(AgentError):
        run_agent("Loop forever", client=client, max_steps=3)
    assert client.calls == 3
