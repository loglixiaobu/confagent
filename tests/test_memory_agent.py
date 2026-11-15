import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from apibank_env import api_schema_to_tool
from memory_agent import RequestBudget, parse_action, run_next_step


TOOL = {
    "name": "HotelAvailability",
    "input_parameters": {"check_in": {"type": "str"}, "check_out": {"type": "str"}},
}


def choice(content="", tool_calls=None):
    return {"message": {"content": content, "tool_calls": tool_calls}, "finish_reason": "stop"}


def tool_call(arguments):
    return {"function": {"name": "HotelAvailability", "arguments": arguments}}


def response(selected=None, prompt_tokens=8, completion_tokens=2):
    return {
        "choices": [selected or choice()],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class MemoryAgentTests(unittest.TestCase):
    def test_cap_consumes_full_prompt_cost_across_requests(self):
        requests = []

        def send(request_id, kind, payload):
            requests.append((request_id, kind, payload))
            return response(prompt_tokens=8, completion_tokens=payload["max_tokens"])

        budget = RequestBudget(20, send, lambda payload: 8)
        budget.complete("first", "test", {"max_tokens": 2})
        budget.complete("second", "test", {"max_tokens": 10})
        self.assertEqual(requests[1][2]["max_tokens"], 2)
        self.assertEqual(budget.cost, {
            "requests": 2, "prompt_tokens": 16, "completion_tokens": 4, "total_tokens": 20,
        })
        with self.assertRaisesRegex(ValueError, "exhausted"):
            budget.complete("third", "test", {"max_tokens": 10})
        self.assertEqual(len(requests), 2)

    def test_rejects_incompatible_prompt_tokenization(self):
        budget = RequestBudget(20, lambda *_: response(prompt_tokens=9), lambda _: 8)
        with self.assertRaisesRegex(ValueError, "Prompt token count mismatch"):
            budget.complete("test", "test", {"max_tokens": 10})

    def test_matched_memory_changes_only_system_prompt_and_charged_tokens(self):
        requests = []
        case = {
            "task_id": "case", "requirement": "Check hotels for November 28.",
            "history": [{"role": "user", "content": "I am visiting Amsterdam."},
                        {"role": "assistant", "content": "Which dates will you visit?"}],
            "api_documents": [TOOL], "expected": "holdout label", "parameter_sources": "private values",
        }
        rules = [
            {
                "rule_id": "applies",
                "applicability": {"problem_type": "missing_required_parameter", "required_fields": ["check_out"]},
                "handling": "Get evidence for the required checkout date before calling.",
                "source_ids": ["training_id"],
            },
            {
                "rule_id": "irrelevant",
                "applicability": {"problem_type": "missing_required_parameter", "tools": ["Calculator"]},
                "handling": "The other rule.",
                "source_ids": ["other_training_id"],
            },
        ]

        def count(payload):
            return len(json.dumps(payload["messages"]).split())

        def send(request_id, kind, payload):
            requests.append(copy.deepcopy(payload))
            return response(choice("What is your checkout date?"), prompt_tokens=count(payload))

        with patch("memory_agent.case_tools", return_value=[api_schema_to_tool(TOOL)]):
            a = run_next_step(case, send=send, count_prompt_tokens=count, request_id="A")
            b = run_next_step(case, send=send, count_prompt_tokens=count, request_id="B", rules=rules)
        self.assertEqual(a["memory_rule_ids"], [])
        self.assertEqual(b["memory_rule_ids"], ["applies"])
        self.assertEqual(a["precheck"]["checks_used"], b["precheck"]["checks_used"])
        self.assertGreater(b["cost"]["prompt_tokens"], a["cost"]["prompt_tokens"])
        self.assertIn(rules[0]["handling"], requests[1]["messages"][0]["content"])
        self.assertNotIn(rules[1]["handling"], requests[1]["messages"][0]["content"])
        self.assertEqual(requests[0]["messages"][1:3], case["history"])
        self.assertEqual(requests[0]["messages"][-1]["content"], "Requirement: " + case["requirement"])
        for private_text in ("holdout label", "private values", "training_id"):
            self.assertNotIn(private_text, json.dumps(requests))
        requests[1]["messages"][0]["content"] = requests[0]["messages"][0]["content"]
        self.assertEqual(requests[0], requests[1])

    def test_parse_multiple_calls_keeps_arguments_for_fabrication_audit(self):
        action = parse_action(choice(tool_calls=[tool_call('{"check_out":"2026-11-30"}'), tool_call("[]")]))
        self.assertEqual(action["error"], "multiple_tool_calls")
        self.assertEqual(action["actions"][0]["arguments"], {"check_out": "2026-11-30"})
        self.assertEqual(action["actions"][1]["error"], "invalid_tool_arguments")

    def test_question_in_hidden_thinking_is_never_graded_as_clarification(self):
        action = parse_action(choice("<think>What is the checkout date?</think>I'll look into it."))
        self.assertEqual(action, {"kind": "other", "text": "I'll look into it."})

    def test_clarification_does_not_require_a_question_mark(self):
        for text in ("I need your checkout date", "Provide the checkout date."):
            self.assertEqual(parse_action(choice(text)), {"kind": "clarify", "text": text})


if __name__ == "__main__":
    unittest.main()
