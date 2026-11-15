"""Regression checks for the acceptance gate and held-out data isolation."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from memory_experiment_data import load_cases
from run_memory_experiment import AgentSettings, build_memory


NO_COST = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class MemoryAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()

    def exercise_gate(self, fail_retry, inject_rules=True):
        validation_attempts, synthesis = [], []
        rule = {
            "rule_id": "verified-arguments",
            "applicability": {
                "problem_type": "missing_required_parameter", "tools": [], "required_fields": [],
            },
            "source_ids": [self.cases["development"][0]["task_id"]],
            "handling": "Check all inputs before calling.",
        }

        def send(request_id, kind, payload):
            self.assertEqual(kind, "memory_synthesis")
            synthesis.append(payload)
            return {"choices": [{"message": {"content": json.dumps({"rules": [rule]})}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}

        def run_next(case, *, request_id, **kwargs):
            parts = request_id.split(":")
            expected = case["expected"]
            if parts[0] == "validation":
                validation_attempts.append(request_id)
            if (inject_rules and parts[0] == "validation" and (parts[1] == "1" or fail_retry)
                    and expected["kind"] == "lookup"):
                action = {"kind": "other", "text": "Done."}
            elif expected["kind"] == "clarify":
                action = {"kind": "clarify", "text": f"Please provide {' and '.join(expected['fields'])}."}
            else:
                action = {"kind": "call", "api_name": expected["api_name"], "arguments": expected["arguments"]}
            return {"action": action, "cost": NO_COST,
                    "memory_rule_ids": [rule["rule_id"] for rule in kwargs["rules"]] if inject_rules else []}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (patch("run_memory_experiment.run_next_step", side_effect=run_next),
                  patch("run_memory_experiment.self_report_for_action",
                        return_value={"self_report": None, "cost": NO_COST}),
                  redirect_stdout(io.StringIO())):
                memory, _, attempts = build_memory(
                    output, self.cases, SimpleNamespace(send=send), lambda payload: 10, AgentSettings(), [],
                )
            self.assertEqual((output / "memory_M1.json").exists(), not fail_retry)
        self.assertEqual(len(synthesis), 2)
        self.assertEqual(len(validation_attempts), 8)
        self.assertEqual(attempts[0]["status"], "failed_validation")
        for payload in synthesis:
            evidence = json.loads(payload["messages"][1]["content"])
            self.assertEqual({row["source_id"] for row in evidence},
                             {case["task_id"] for case in self.cases["development"]})
        return memory, attempts

    def test_second_candidate_can_pass_only_after_all_four_validations(self):
        memory, attempts = self.exercise_gate(fail_retry=False)
        self.assertEqual(memory["accepted_attempt"], 2)
        self.assertEqual(attempts[1]["status"], "passed")

    def test_second_failure_does_not_create_m1(self):
        memory, attempts = self.exercise_gate(fail_retry=True)
        self.assertIsNone(memory)
        self.assertEqual(attempts[1]["status"], "failed_validation")

    def test_rules_that_were_never_injected_are_not_accepted(self):
        memory, attempts = self.exercise_gate(fail_retry=True, inject_rules=False)
        self.assertIsNone(memory)
        for attempt in attempts:
            self.assertTrue(all(row["evaluation"]["correct"] for row in attempt["validation"]))
            self.assertEqual(attempt["rule_coverage"], {"verified-arguments": []})
