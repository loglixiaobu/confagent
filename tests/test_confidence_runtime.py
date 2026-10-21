"""CPU checks for the deployable consistency confidence API; no model calls."""
import ast
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import confidence_runtime as runtime


def calibrator():
    return {"version": "consistency-platt-v1", "signal": "consistency", "link": "sigmoid",
            "coefficient": 4.0, "intercept": -2.0, "input_range": [0, 1],
            "score_direction": "higher_is_more_correct",
            "decision_rule": {"signal": "consistency", "operator": "<", "threshold": 1.0},
            "collection_protocol": {"consistency_k": 3, "consistency_thinking": False,
                                    "consistency_temperature": .7, "consistency_top_p": 1.,
                                    "max_tokens": 4096}}


def action(identifier="CANDIDATE_ONLY_SENTINEL"):
    return {"api_name": "Lookup", "arguments": {"id": identifier}}


def state():
    return {"messages": [{"role": "system", "content": "Choose one tool."},
                         {"role": "user", "content": "Retrieve the requested record."}],
            "tools": [{"type": "function", "function": {"name": "Lookup", "parameters": {
                "type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}}],
            "candidate": action(),
            "api_documents": [{"name": "Lookup", "input_parameters": {"id": {"type": "str"}}}]}


def response(candidate=None, *, finish_reason="tool_calls", extra_calls=(), content=None):
    candidate = candidate if candidate is not None else action()
    calls = [] if candidate is False else [{"id": "call_0", "type": "function", "function": {
        "name": candidate["api_name"], "arguments": json.dumps(candidate["arguments"])}}]
    calls.extend(extra_calls)
    return {"choices": [{"message": {"content": content, "tool_calls": calls},
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class CalibratorTests(unittest.TestCase):
    def test_sigmoid_mapping_and_raw_verification_rule(self):
        for raw in (0, 1 / 3, 2 / 3, 1):
            result = runtime.calibrate_consistency(raw, calibrator())
            self.assertAlmostEqual(result["p_correct"], 1 / (1 + math.exp(2 - 4 * raw)))
            self.assertEqual(result["needs_verification"], raw < 1)
        # A prior-only fallback may still use raw agreement for verification.
        prior = dict(calibrator(), coefficient=0., intercept=1.)
        self.assertEqual(runtime.calibrate_consistency(0, prior)["p_correct"],
                         runtime.calibrate_consistency(1, prior)["p_correct"])
        self.assertTrue(runtime.calibrate_consistency(0, prior)["needs_verification"])
        self.assertFalse(runtime.calibrate_consistency(1, prior)["needs_verification"])

    def test_missing_is_not_zero_and_extreme_logits_are_stable(self):
        missing = runtime.calibrate_consistency(None, calibrator())
        self.assertIsNone(missing["raw_consistency"])
        self.assertIsNone(missing["p_correct"])
        self.assertTrue(missing["needs_verification"])
        self.assertEqual(missing["reason"], "score_missing")
        for intercept, expected in ((10000., 1.), (-10000., 0.)):
            self.assertEqual(runtime.calibrate_consistency(0, dict(calibrator(), intercept=intercept))["p_correct"], expected)

    def test_bad_raw_values_and_incompatible_artifacts_rejected(self):
        for value in (True, False, float("nan"), float("inf"), -.01, 1.01, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.calibrate_consistency(value, calibrator())
        changes = [{"version": "unknown"}, {"signal": "self_report"}, {"link": "identity"},
                   {"score_direction": "lower_is_more_correct"}, {"coefficient": -1.},
                   {"coefficient": True}, {"intercept": float("nan")}, {"input_range": [False, True]},
                   {"decision_rule": {"signal": "consistency", "operator": "<=", "threshold": 1}},
                   {"collection_protocol": dict(calibrator()["collection_protocol"], consistency_k=True)},
                   {"collection_protocol": dict(calibrator()["collection_protocol"], consistency_thinking=True)}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                runtime.load_calibrator(dict(calibrator(), **change))

    def test_offline_cli_needs_no_site_packages_or_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibrator.json"
            path.write_text(json.dumps(calibrator()))
            completed = subprocess.run([sys.executable, "-S", str(ROOT / "code/confidence_runtime.py"),
                                        "--calibrator", str(path), "--raw-consistency", "1"],
                                       capture_output=True, text=True, check=True)
        result = json.loads(completed.stdout)
        self.assertAlmostEqual(result["p_correct"], 1 / (1 + math.exp(-2)))
        self.assertFalse(result["needs_verification"])


class RuntimeTests(unittest.TestCase):
    def estimate(self, request_fn, **kwargs):
        with patch("confidence_runtime.time.sleep") as sleep:
            result = runtime.estimate_step_confidence(**state(), calibrator=calibrator(),
                                                      request_fn=request_fn, **kwargs)
        for call in sleep.call_args_list:
            self.assertGreaterEqual(call.args[0], 1)
        json.dumps(result, allow_nan=False)
        return result

    def test_candidate_is_never_added_to_requests_and_inputs_are_unchanged(self):
        inputs = state()
        before = copy.deepcopy(inputs)
        request_fn = Mock(side_effect=[response() for _ in range(3)])
        with patch("confidence_runtime.time.sleep") as sleep:
            result = runtime.estimate_step_confidence(**inputs, calibrator=calibrator(), request_fn=request_fn,
                                                      request_namespace="task:step", seed=123)
        self.assertEqual(inputs, before)
        self.assertEqual(request_fn.call_count, 3)
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(len(set(call.kwargs["seed"] for call in request_fn.call_args_list)), 3)
        for index, call in enumerate(request_fn.call_args_list):
            payload = call.kwargs
            self.assertEqual(payload["messages"], inputs["messages"])
            self.assertEqual(payload["tools"], inputs["tools"])
            self.assertNotIn("CANDIDATE_ONLY_SENTINEL", json.dumps(payload))
            self.assertEqual(payload["model"], "Qwen3-8B")
            self.assertEqual(payload["temperature"], .7)
            self.assertEqual(payload["top_p"], 1.)
            self.assertEqual(payload["max_tokens"], 4096)
            self.assertIs(payload["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)
            self.assertEqual(payload["seed"], 123 + int(runtime.digest(f"task:step:sample_{index}")[:7], 16))
            self.assertEqual(result["samples"][index]["payload_sha256"], runtime.digest(payload))
        self.assertEqual(result["raw_consistency"], 1)
        self.assertFalse(result["needs_verification"])
        self.assertEqual(result["cost"]["usage"]["total_tokens"], 45)
        self.assertEqual(result["quality"]["n_samples"], 3)

    def test_no_tool_is_disagreement_and_not_a_missing_response(self):
        result = self.estimate(Mock(side_effect=[response(), response(action("OTHER")),
                                                response(False, finish_reason="stop", content="Done.")]))
        self.assertEqual(result["raw_consistency"], 1 / 3)
        self.assertIsNotNone(result["p_correct"])
        self.assertTrue(result["needs_verification"])
        self.assertEqual(result["quality"]["n_no_tool"], 1)
        self.assertTrue(result["quality"]["complete"])

    def test_truncation_and_multiple_calls_keep_the_frozen_first_call_rule(self):
        extra = response(action("SECOND"))["choices"][0]["message"]["tool_calls"]
        result = self.estimate(Mock(side_effect=[response(extra_calls=extra),
                                                response(finish_reason="length"), response()]))
        self.assertEqual(result["raw_consistency"], 1)
        self.assertEqual(result["quality"]["n_multiple_tool_calls"], 1)
        self.assertEqual(result["quality"]["n_truncated"], 1)
        self.assertEqual(result["samples"][0]["n_tool_calls"], 2)
        self.assertEqual(result["samples"][0]["action"], action())

    def test_transport_failure_retries_same_seed_and_can_recover(self):
        request_fn = Mock(side_effect=[TimeoutError("temporary"), response(), response(), response()])
        result = self.estimate(request_fn)
        self.assertEqual(request_fn.call_args_list[0], request_fn.call_args_list[1])
        self.assertEqual(result["raw_consistency"], 1)
        self.assertEqual(result["cost"]["n_model_requests"], 4)
        self.assertEqual(result["quality"]["n_transport_failures"], 1)

    def test_missing_sample_has_no_fake_probability_or_renormalized_score(self):
        result = self.estimate(Mock(side_effect=[response(), TimeoutError("one"), TimeoutError("two"), response()]))
        self.assertIsNone(result["raw_consistency"])
        self.assertIsNone(result["p_correct"])
        self.assertTrue(result["needs_verification"])
        self.assertEqual(result["reason"], "score_missing")
        self.assertEqual(result["quality"]["n_samples"], 2)
        self.assertEqual(result["quality"]["n_agree"], 2)
        self.assertEqual(result["cost"]["n_model_requests"], 4)

    def test_invalid_response_is_missing_but_invalid_argument_json_is_disagreement(self):
        invalid_call = response()
        invalid_call["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{broken"
        result = self.estimate(Mock(side_effect=[invalid_call, response(), response()]))
        self.assertEqual(result["raw_consistency"], 2 / 3)
        self.assertEqual(result["quality"]["n_invalid_tool_arguments"], 1)
        missing = self.estimate(Mock(side_effect=[{}, {}, response(), response()]))
        self.assertIsNone(missing["p_correct"])
        self.assertEqual(missing["quality"]["n_invalid_responses"], 2)

    def test_gates_abort_further_requests_and_do_not_fill_missing_samples(self):
        request_fn = Mock(return_value=response())
        blocked = self.estimate(request_fn, before_request=lambda: False)
        self.assertEqual(request_fn.call_count, 0)
        self.assertEqual(blocked["cost"]["n_model_requests"], 0)
        self.assertIsNone(blocked["p_correct"])
        self.assertEqual(blocked["quality"]["n_gate_blocks"], 1)
        request_fn = Mock(return_value=response())
        halted = self.estimate(request_fn, after_request=lambda _: False)
        self.assertEqual(request_fn.call_count, 1)
        self.assertEqual(halted["quality"]["n_samples"], 1)
        self.assertIsNone(halted["p_correct"])

    def test_injected_client_is_used_without_being_closed_by_runtime(self):
        create = Mock(side_effect=[SimpleNamespace(model_dump=lambda: response()) for _ in range(3)])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=Mock())
        with patch("confidence_runtime.time.sleep"):
            result = runtime.estimate_step_confidence(**state(), calibrator=calibrator(), client=client)
        self.assertEqual(result["raw_consistency"], 1)
        client.close.assert_not_called()

    def test_invalid_runtime_settings_fail_before_inference(self):
        for change in ({"cooldown_s": .5}, {"cooldown_s": float("nan")}, {"seed": True},
                       {"messages": []}, {"candidate": {}}, {"api_documents": None}):
            kwargs = dict(state(), calibrator=calibrator(), request_fn=Mock())
            kwargs.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                runtime.estimate_step_confidence(**kwargs)
            kwargs["request_fn"].assert_not_called()

    def test_semantic_helpers_remain_identical_to_frozen_collector(self):
        old = ast.parse((ROOT / "code/collect_new_signals.py").read_text())
        new = ast.parse((ROOT / "code/confidence_runtime.py").read_text())
        for name in ("digest", "equivalent", "sample_action"):
            original = next(node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == name)
            current = next(node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(current, include_attributes=False))


if __name__ == "__main__":
    unittest.main()
