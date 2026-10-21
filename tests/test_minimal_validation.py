"""CPU checks for frozen new-task validation, coverage and allocation metrics."""
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from confidence_runtime import calibrate_consistency
from evaluate_minimal_validation import (budget_metrics, evaluate_rows, join_validation,
                                         probability_metrics, report, row_key, run_diagnostics, verification_metrics)


def calibrator():
    return {"version": "consistency-platt-v1", "signal": "consistency", "link": "sigmoid",
            "coefficient": 4., "intercept": -2., "input_range": [0, 1],
            "score_direction": "higher_is_more_correct", "training_prior": .8,
            "decision_rule": {"signal": "consistency", "operator": "<", "threshold": 1.},
            "collection_protocol": {"consistency_k": 3, "consistency_thinking": False,
                                    "consistency_temperature": .7, "consistency_top_p": 1., "max_tokens": 4096}}


def fixture():
    steps, labels = [], []
    for index, (raw, y) in enumerate(((0., 0), (1 / 3, 0), (2 / 3, 1), (1., 1), (None, 0)), 1):
        steps.append({"task_id": "new-task", "step_id": index,
                      "action": {"api_name": "Lookup", "arguments": {"id": index}},
                      "confidence": dict(calibrate_consistency(raw, calibrator()), quality={}),
                      "raw_logprobs": {"available": index != 2, "geo_mean_prob": index / 5}})
        labels.append({"key": f"new-task:{index}", "y": y, "reason": "fixture independent label"})
    return steps, labels


class ValidationTests(unittest.TestCase):
    def test_strict_join_mapping_and_separate_coverage(self):
        steps, labels = fixture()
        rows, quality = join_validation(steps, labels, calibrator())
        metrics = evaluate_rows(rows, quality)
        available = metrics["cohorts"]["all_steps"]["signals"]
        self.assertEqual(metrics["n_steps"], 5)
        self.assertEqual(metrics["positive"], 2)
        self.assertEqual(metrics["negative"], 3)
        self.assertEqual(available["raw_consistency"]["n"], 4)
        self.assertEqual(available["p_correct"]["n"], 4)
        self.assertEqual(available["training_prior"]["n"], 5)
        self.assertEqual(available["logprob"]["n"], 4)
        self.assertEqual(metrics["cohorts"]["matched_all_signals"]["n"], 3)
        self.assertEqual(metrics["cohorts"]["matched_consistency"]["n"], 4)
        self.assertEqual(available["raw_consistency"]["AUROC"], 1)
        self.assertEqual(available["p_correct"]["AUROC"], 1)
        self.assertEqual(available["training_prior"]["AUROC"], .5)
        self.assertEqual(quality["missing_consistency"], 1)
        json.dumps(metrics, allow_nan=False)

    def test_missing_duplicate_nonbinary_mismatched_probability_and_policy_rejected(self):
        steps, labels = fixture()
        for bad_steps, bad_labels in ((steps + [steps[0]], labels), (steps, labels + [labels[0]]),
                                      (steps, labels[:-1]), (steps, [dict(labels[0], y=True)] + labels[1:])):
            with self.assertRaises(ValueError):
                join_validation(bad_steps, bad_labels, calibrator())
        for field, value in (("p_correct", .123), ("raw_consistency", .5),
                             ("needs_verification", False), ("p_correct", None)):
            changed = copy.deepcopy(steps)
            changed[0]["confidence"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                join_validation(changed, labels, calibrator())
        changed = copy.deepcopy(steps)
        changed[-1]["confidence"]["p_correct"] = .5
        with self.assertRaises(ValueError):
            join_validation(changed, labels, calibrator())

    def test_identifier_variants_must_agree(self):
        self.assertEqual(row_key({"key": "task:1"}), ("task", 1))
        self.assertEqual(row_key({"key": ["task", 1]}), ("task", 1))
        self.assertEqual(row_key({"key": "task:1", "task_id": "task", "step_id": 1}), ("task", 1))
        for row in ({"key": "task:bad"}, {"key": ["task", True]}, {"key": "task:1", "task_id": "task", "step_id": 2}):
            with self.assertRaises(ValueError):
                row_key(row)

    def test_probability_metrics_handle_single_class_no_coverage_and_probability_edges(self):
        rows = [{"y": 1, "score": 1.}, {"y": 1, "score": 0.}]
        metrics = probability_metrics(rows, "score")
        self.assertIsNone(metrics["AUROC"])
        self.assertEqual(metrics["AUROC_unavailable_reason"], "single_class")
        self.assertEqual(metrics["Brier"], .5)
        self.assertEqual(metrics["ECE"], .5)
        self.assertTrue(math.isfinite(metrics["NLL"]))
        self.assertEqual(sum(row["n"] for row in metrics["reliability_bins"]), 2)
        missing = probability_metrics([{"y": 0, "score": None}], "score")
        self.assertIsNone(missing["AUROC"])
        self.assertEqual(missing["AUROC_unavailable_reason"], "no_scored_steps")
        self.assertIsNone(missing["Brier"])

    def test_budget_is_strict_deterministic_label_blind_and_reports_tie_expectation(self):
        rows = [{"task_id": "task", "step_id": index + 1, "y": index % 2, "score": .5}
                for index in range(10)]
        metrics = budget_metrics(rows, "score")
        self.assertEqual(metrics["budget"], 2)
        self.assertEqual(metrics["captured_errors_random_tie_expectation"], 1)
        self.assertEqual(metrics["error_capture_rate_random_tie_expectation"], .2)
        self.assertEqual(metrics, budget_metrics(list(reversed(rows)), "score"))
        changed = [dict(row, y=1 - row["y"]) for row in rows]
        self.assertEqual(metrics["selected_keys"], budget_metrics(changed, "score")["selected_keys"])
        self.assertEqual(budget_metrics(rows[:4], "score")["budget"], 0)

    def test_missing_scores_trigger_verification_and_are_explicit_in_all_step_budget(self):
        steps, labels = fixture()
        rows, _ = join_validation(steps, labels, calibrator())
        fixed = verification_metrics(rows)
        self.assertEqual(fixed["selected"], 4)
        self.assertEqual(fixed["captured_errors"], 3)
        self.assertEqual(fixed["error_capture_rate"], 1)
        self.assertEqual(fixed["missing_score_selected"], 1)
        all_steps = budget_metrics(rows, "raw_consistency", missing_first=True)
        self.assertEqual(all_steps["budget"], 1)
        self.assertEqual(all_steps["selected_keys"], ["new-task:5"])
        self.assertIsNone(verification_metrics([dict(rows[0], y=1)])["error_capture_rate"])

    def test_end_to_end_writes_metrics_plot_report_without_modifying_inputs(self):
        steps, labels = fixture()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            inputs = {"calibrator.json": json.dumps(calibrator()) + "\n",
                      "validation_protocol.json": json.dumps({"frozen": True, "scope": "fixture"}) + "\n",
                      "validation_steps.jsonl": "".join(json.dumps(row) + "\n" for row in steps),
                      "validation_labels.jsonl": "".join(json.dumps(row) + "\n" for row in labels)}
            for name, text in inputs.items():
                (out / name).write_text(text)
            completed = subprocess.run([sys.executable, str(ROOT / "code/evaluate_minimal_validation.py"), "--out-dir", folder],
                                       capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(completed.stdout)["steps"], 5)
            self.assertTrue((out / "validation_metrics.json").is_file())
            self.assertGreater((out / "validation_diagnostic.png").stat().st_size, 1000)
            report = (out / "新任务验证报告.md").read_text()
            self.assertIn("不能解释为大规模独立泛化证明", report)
            self.assertNotIn(folder, report)
            for name, text in inputs.items():
                self.assertEqual((out / name).read_text(), text)

    def test_actual_task_and_cost_diagnostics_include_zero_call_tasks(self):
        steps, labels = fixture()
        rows, quality = join_validation(steps, labels, calibrator())
        metrics = evaluate_rows(rows, quality)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            trajectories = [{"task_id": "new-task", "steps": 5, "stop_reason": "final_answer",
                             "real_tool_executions": 0, "recommendations_applied": False},
                            {"task_id": "zero-step-task", "steps": 0, "stop_reason": "no_tool",
                             "real_tool_executions": 0, "recommendations_applied": False}]
            (out / "validation_trajectories.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trajectories))
            marker = {"complete": True, "calibrator_unchanged": True, "tasks": 2, "steps": 5, "correct": 2, "errors": 3}
            (out / "validation_complete.json").write_text(json.dumps(marker))
            request = {"kind": "confidence_probe", "latency_s": .5,
                       "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}}
            (out / "validation_requests.jsonl").write_text(json.dumps(request) + "\n")
            result = run_diagnostics(out, metrics, rows)
            self.assertEqual(result["executed_tasks"], 2)
            self.assertEqual(result["zero_call_tasks"], 1)
            self.assertFalse(result["task_success_evaluated"])
            self.assertEqual(result["model_cost"]["total_tokens_with_usage"], 15)
            (out / "validation_complete.json").write_text(json.dumps(dict(marker, errors=4)))
            with self.assertRaises(ValueError):
                run_diagnostics(out, metrics, rows)

    def test_optional_task_outcomes_align_and_binary_labels_are_strict(self):
        steps, labels = fixture()
        rows, quality = join_validation(steps, labels, calibrator())
        metrics = evaluate_rows(rows, quality)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            trajectories = [{"task_id": "new-task", "steps": 5}, {"task_id": "zero-task", "steps": 0}]
            (out / "validation_trajectories.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trajectories))
            path = out / "validation_task_outcomes.jsonl"
            outcomes = [{"task_id": "new-task", "task_success": 1, "stop_event_y": 1, "reason_code": "complete"},
                        {"task_id": "zero-task", "task_success": 0, "stop_event_y": 0, "reason_code": "generation_truncated"}]
            path.write_text("".join(json.dumps(row) + "\n" for row in outcomes))
            result = run_diagnostics(out, metrics, rows)
            self.assertTrue(result["task_success_evaluated"])
            self.assertEqual(result["task_outcomes"]["successful_tasks"], 1)
            self.assertEqual(result["task_outcomes"]["failed_tasks"], 1)
            self.assertEqual(result["task_outcomes"]["failure_reason_codes"], {"generation_truncated": 1})
            bad_inputs = [outcomes[:-1], outcomes + [outcomes[0]],
                          [dict(outcomes[0], task_success=True), outcomes[1]],
                          [outcomes[0], dict(outcomes[1], stop_event_y=None)]]
            for bad in bad_inputs:
                path.write_text("".join(json.dumps(row) + "\n" for row in bad))
                with self.assertRaises(ValueError):
                    run_diagnostics(out, metrics, rows)

    def test_report_warns_when_errors_are_few_and_calibration_loses_to_prior(self):
        steps, labels = fixture()
        for index, step in enumerate(steps):
            step["confidence"] = dict(calibrate_consistency(0., calibrator()), quality={})
            labels[index]["y"] = 0 if index == 0 else 1
        rows, quality = join_validation(steps, labels, calibrator())
        metrics = evaluate_rows(rows, quality)
        self.assertTrue(metrics["limitations"]["calibrated_brier_exceeds_training_prior"])
        metrics["run_diagnostics"] = {"task_success_evaluated": True,
                                      "task_outcomes": {"n_tasks": 1, "successful_tasks": 0, "failed_tasks": 1,
                                                        "task_success_rate": 0., "correct_stop_events": 0,
                                                        "incorrect_stop_events": 1, "failure_reason_codes": {"generation_truncated": 1}}}
        with tempfile.TemporaryDirectory() as folder:
            report(Path(folder), metrics)
            text = (Path(folder) / "新任务验证报告.md").read_text()
        self.assertIn("负例仅 1 个", text)
        self.assertIn("不能声称新域概率优于常数基线", text)
        self.assertIn("不能与此前旧 50 个任务的成功率直接比较", text)


if __name__ == "__main__":
    unittest.main()
