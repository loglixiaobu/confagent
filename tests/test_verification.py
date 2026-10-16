import json
import os
import sys
import unittest
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from apibank_env import APIBankMockEnv, api_schema_to_tool  # noqa: E402
from action_semantics import semantic_action_equal  # noqa: E402
from react_agent import AgentConfig, run_task  # noqa: E402
from verification import (BudgetState, CandidateAction,  # noqa: E402
                          check_argument_provenance, correction_outcome,
                          grounded_label, run_consistency_vote, run_critic,
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
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


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
    def test_online_consistency_uses_published_semantic_equality(self):
        calculator = {
            "name": "Calculator", "description": "Calculate.",
            "input_parameters": {"formula": {"type": "str", "description": "Formula."}},
        }
        tools = [api_schema_to_tool(calculator)]
        self.assertTrue(semantic_action_equal(
            CandidateAction("Calculator", {"formula": "70 + 80 + 90"}),
            CandidateAction("Calculator", {"formula": "240"}), tools))
        self.assertFalse(semantic_action_equal(
            CandidateAction("Calculator", {"formula": "70 + 80"}),
            CandidateAction("Calculator", {"formula": "240"}), tools))

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

    def test_provenance_accepts_visible_number_and_reformatted_datetime(self):
        api = {
            "name": "ReserveRoom", "description": "Reserve a room.",
            "input_parameters": {
                "room": {"type": "str", "description": "Room name."},
                "people": {"type": "int", "description": "Party size."},
                "start_time": {"type": "str", "description": "Start date and time."},
            },
        }
        result = check_argument_provenance(
            "Reserve room Atlas for 4 people on 2026-11-03 at 14:00.", [],
            CandidateAction("ReserveRoom", {
                "room": "Atlas", "people": 4, "start_time": "2026-11-03T14:00",
            }), [api_schema_to_tool(api)])
        self.assertTrue(result.valid)
        self.assertEqual(result.supported_fields, ["room", "people", "start_time"])

    def test_critic_json_without_boolean_valid_is_unavailable(self):
        task, api = task_fixture()
        tools = [api_schema_to_tool(api)]
        candidate = CandidateAction("LookupUser", {"user_name": "Alice"})
        provenance = check_argument_provenance(task["requirement"], [], candidate, tools)
        client = FakeClient([text_response(json.dumps({
            "candidate_action": candidate.to_dict(), "revised_action": None,
        }))])
        result = run_critic(client, "fake", [], tools, candidate, provenance)
        self.assertFalse(result.available)
        self.assertIsNone(result.valid)
        self.assertEqual(result.issues[0]["kind"], "invalid_critic_output")

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
        hit, reason = should_verify("always", 1.0, prov, BudgetState(2), 0.0,
                                    __import__("random").Random(0), required_calls=3)
        self.assertFalse(hit)
        self.assertIn("exhausted", reason)

    def test_correction_outcomes(self):
        self.assertEqual(correction_outcome(False, True, True), "error_to_correct")
        self.assertEqual(correction_outcome(True, False, True), "correct_to_error")

    def test_grounded_label_rejects_lucky_hidden_value(self):
        task, api = task_fixture()
        provenance = check_argument_provenance(
            task["requirement"], [], CandidateAction("LookupUser", {"user_name": "Bob"}),
            [api_schema_to_tool(api)])
        self.assertFalse(grounded_label(provenance, replay_correct=True))

    def test_consistency_keeps_fixed_denominator_and_missing_is_unavailable(self):
        task, api = task_fixture()
        tools = [api_schema_to_tool(api)]
        candidate = CandidateAction("LookupUser", {"user_name": "Alice"})
        client = FakeClient([
            tool_response("LookupUser", {"user_name": "Alice"}),
            text_response("no action"), text_response("no action"),
        ])
        vote = run_consistency_vote(
            client, "fake", [], tools, candidate, 3, 0.7, 100, False, seed=0)
        self.assertTrue(vote.available)
        self.assertAlmostEqual(vote.raw["agreement"], 1 / 3)
        self.assertEqual(vote.raw["n_samples"], 3)

        missing_client = FakeClient([
            tool_response("LookupUser", {"user_name": "Alice"}),
            RuntimeError("transport"), text_response("no action"),
        ])
        missing = run_consistency_vote(
            missing_client, "fake", [], tools, candidate, 3, 0.7, 100, False, seed=0)
        self.assertFalse(missing.available)
        self.assertIsNone(missing.raw["agreement"])


class AgentFlowTests(unittest.TestCase):
    def test_pre_execution_critic_repairs_candidate(self):
        task, _ = task_fixture()
        critic = text_response(json.dumps({
            "valid": False,
            "issues": [{"field": "user_name", "reason": "unsupported"}],
            "revised_action": {"api_name": "LookupUser", "arguments": {"user_name": "Alice"}},
        }))
        client = FakeClient([tool_response("LookupUser", {"user_name": "Bob"}), critic,
                             text_response("Alice was found.")])
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
        self.assertFalse(step.candidate_grounded_correct)
        self.assertTrue(step.executed_grounded_correct)
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
            text_response("Alice was found."),
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
            text_response("Alice was found."),
        ])
        cfg = AgentConfig(
            model="fake", max_steps=3, verify_policy="exception", verifier="critic",
            verify_budget=0, feedback_mode="realistic", enable_thinking=False,
        )
        traj = run_task(task, cfg, client, task_index=0)
        self.assertTrue(traj.final_success)
        self.assertFalse(traj.clean_success)
        self.assertTrue(traj.recovered_success)
        self.assertEqual([s.action_type for s in traj.steps], ["tool_call", "retry", "answer"])
        self.assertFalse(traj.steps[0].matched_gold)
        self.assertTrue(traj.steps[1].matched_gold)
        self.assertEqual(traj.steps[1].correction_type, "local_retry")
        self.assertEqual(traj.steps[1].correction_outcome, "unchanged_correct")
        self.assertEqual(traj.steps[1].retry_outcome, "recovered")
        self.assertEqual(traj.steps[1].parent_step_id, traj.steps[0].step_id)

    def test_replay_completion_without_final_answer_is_not_task_success(self):
        task, _ = task_fixture()
        client = FakeClient([tool_response("LookupUser", {"user_name": "Alice"})])
        cfg = AgentConfig(model="fake", max_steps=1, verify_policy="vanilla",
                          feedback_mode="realistic", enable_thinking=False)
        traj = run_task(task, cfg, client, task_index=0)
        self.assertTrue(traj.replay_success)
        self.assertFalse(traj.final_success)


if __name__ == "__main__":
    unittest.main()
