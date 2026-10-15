import json
import os
import sys
import unittest
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from apibank_env import APIBankMockEnv, api_schema_to_tool  # noqa: E402
from react_agent import AgentConfig, run_task  # noqa: E402
from verification import (BudgetState, CandidateAction,  # noqa: E402
                          check_argument_provenance, correction_outcome,
                          should_verify)


def task_fixture():
    api = {
        "name": "LookupUser",
        "description": "Look up one user by name.",
        "input_parameters": {
            "user_name": {"type": "str", "description": "Exact user name."}
        },
    }
    return {
        "requirement": "Look up the user Alice.",
        "response": "Alice was found.",
        "apis": [
            {"api_name": "ToolSearcher", "input": {"keywords": "lookup user"},
             "output": {"output": api}, "exception": None},
            {"api_name": "LookupUser", "input": {"user_name": "Alice"},
             "output": {"output": {"id": 7, "name": "Alice"}}, "exception": None},
        ],
    }, api


def tool_response(name, args, call_id="call-1"):
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    tc = SimpleNamespace(id=call_id, function=fn)
    msg = SimpleNamespace(content="<think>choose tool</think>", tool_calls=[tc])
    choice = SimpleNamespace(message=msg, logprobs=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


def text_response(text):
    msg = SimpleNamespace(content=text, tool_calls=[])
    choice = SimpleNamespace(message=msg, logprobs=None)
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
    return SimpleNamespace(choices=[choice], usage=usage)


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected model request")
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class EnvironmentTests(unittest.TestCase):
    def test_failed_then_correct_is_recovered_success(self):
        task, _ = task_fixture()
        env = APIBankMockEnv(task, feedback_mode="realistic")
        bad = env.call("LookupUser", {"user_name": "Bob"})
        self.assertFalse(bad.matched_gold)
        self.assertNotIn("user_name", bad.exception)
        good = env.call("LookupUser", {"user_name": "Alice"})
        self.assertTrue(good.matched_gold)
        self.assertTrue(env.success())
        self.assertFalse(env.clean_success())
        self.assertTrue(env.recovered_success())

    def test_diagnostic_feedback_is_explicit_but_realistic_is_not(self):
        task, _ = task_fixture()
        diagnostic = APIBankMockEnv(task, feedback_mode="diagnostic")
        realistic = APIBankMockEnv(task, feedback_mode="realistic")
        self.assertIn("user_name", diagnostic.call("LookupUser", {"user_name": "Bob"}).exception)
        self.assertNotIn("user_name", realistic.call("LookupUser", {"user_name": "Bob"}).exception)


class VerificationUnitTests(unittest.TestCase):
    def test_provenance_flags_unsupported_value(self):
        task, api = task_fixture()
        tools = [api_schema_to_tool(api)]
        bad = check_argument_provenance(
            task["requirement"], [], CandidateAction("LookupUser", {"user_name": "Bob"}), tools)
        good = check_argument_provenance(
            task["requirement"], [], CandidateAction("LookupUser", {"user_name": "Alice"}), tools)
        self.assertGreaterEqual(bad.risk, 0.7)
        self.assertFalse(bad.valid)
        self.assertTrue(good.valid)

    def test_budget_and_policy(self):
        task, api = task_fixture()
        prov = check_argument_provenance(
            task["requirement"], [], CandidateAction("LookupUser", {"user_name": "Bob"}),
            [api_schema_to_tool(api)])
        budget = BudgetState(1)
        hit, _ = should_verify("provenance", prov.risk, prov, budget, 0.6,
                               __import__("random").Random(0))
        self.assertTrue(hit)
        self.assertTrue(budget.spend(1))
        hit, reason = should_verify("always", 1.0, prov, budget, 0.0,
                                    __import__("random").Random(0))
        self.assertFalse(hit)
        self.assertIn("exhausted", reason)

    def test_correction_outcomes(self):
        self.assertEqual(correction_outcome(False, True, True), "error_to_correct")
        self.assertEqual(correction_outcome(True, False, True), "correct_to_error")


class AgentFlowTests(unittest.TestCase):
    def test_pre_execution_critic_repairs_candidate(self):
        task, _ = task_fixture()
        critic = text_response(json.dumps({
            "valid": False,
            "issues": [{"field": "user_name", "reason": "unsupported"}],
            "revised_action": {"api_name": "LookupUser", "arguments": {"user_name": "Alice"}},
        }))
        client = FakeClient([tool_response("LookupUser", {"user_name": "Bob"}), critic])
        cfg = AgentConfig(
            model="fake", max_steps=2, verify_policy="provenance", verifier="critic",
            verify_budget=1, feedback_mode="realistic", enable_thinking=False,
        )
        traj = run_task(task, cfg, client, task_index=0)
        self.assertTrue(traj.final_success)
        self.assertTrue(traj.clean_success)
        self.assertFalse(traj.recovered_success)
        step = traj.steps[0]
        self.assertTrue(step.verified)
        self.assertFalse(step.candidate_matched_gold)
        self.assertTrue(step.matched_gold)
        self.assertEqual(step.correction_outcome, "error_to_correct")
        self.assertEqual(step.action_input["arguments"]["user_name"], "Alice")
        self.assertEqual(step.model_calls, 2)
        self.assertEqual(step.verification_model_calls, 1)
        self.assertEqual(traj.config["verification_budget_used"], 1)

    def test_consistency_vote_uses_budget_and_majority_action(self):
        task, _ = task_fixture()
        client = FakeClient([
            tool_response("LookupUser", {"user_name": "Bob"}, "base"),
            tool_response("LookupUser", {"user_name": "Alice"}, "s1"),
            tool_response("LookupUser", {"user_name": "Alice"}, "s2"),
            tool_response("LookupUser", {"user_name": "Bob"}, "s3"),
        ])
        cfg = AgentConfig(
            model="fake", max_steps=2, verify_policy="provenance", verifier="consistency",
            verify_budget=3, consistency_k=3, feedback_mode="realistic",
            enable_thinking=False,
        )
        traj = run_task(task, cfg, client, task_index=0)
        step = traj.steps[0]
        self.assertTrue(traj.final_success)
        self.assertEqual(step.action_input["arguments"]["user_name"], "Alice")
        self.assertEqual(step.verification_model_calls, 3)
        self.assertEqual(traj.config["verification_budget_used"], 3)

    def test_harmful_correction_is_visible(self):
        task, _ = task_fixture()
        critic = text_response(json.dumps({
            "valid": False,
            "issues": [{"field": "user_name", "reason": "incorrect critique"}],
            "revised_action": {"api_name": "LookupUser", "arguments": {"user_name": "Bob"}},
        }))
        client = FakeClient([tool_response("LookupUser", {"user_name": "Alice"}), critic])
        cfg = AgentConfig(
            model="fake", max_steps=1, verify_policy="always", verifier="critic",
            verify_budget=1, feedback_mode="realistic", enable_thinking=False,
        )
        traj = run_task(task, cfg, client, task_index=0)
        self.assertFalse(traj.final_success)
        self.assertEqual(traj.steps[0].correction_outcome, "correct_to_error")

    def test_environment_feedback_drives_local_retry(self):
        task, _ = task_fixture()
        client = FakeClient([
            tool_response("LookupUser", {"user_name": "Bob"}, "call-1"),
            tool_response("LookupUser", {"user_name": "Alice"}, "call-2"),
        ])
        cfg = AgentConfig(
            model="fake", max_steps=3, verify_policy="exception", verifier="critic",
            verify_budget=0, feedback_mode="realistic", enable_thinking=False,
        )
        traj = run_task(task, cfg, client, task_index=0)
        self.assertTrue(traj.final_success)
        self.assertFalse(traj.clean_success)
        self.assertTrue(traj.recovered_success)
        self.assertEqual([s.action_type for s in traj.steps], ["tool_call", "retry"])
        self.assertFalse(traj.steps[0].matched_gold)
        self.assertTrue(traj.steps[1].matched_gold)
        self.assertEqual(traj.steps[1].correction_type, "local_retry")
        self.assertEqual(traj.steps[1].correction_outcome, "error_to_correct")
        self.assertEqual(traj.steps[1].parent_step_id, traj.steps[0].step_id)


if __name__ == "__main__":
    unittest.main()
