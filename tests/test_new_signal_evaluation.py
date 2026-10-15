import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from evaluate_new_signals import (collection_diagnostics, error_capture, error_type_diagnostics, evaluate, evaluate_rows, join_scores,
                                  metrics, paired_bootstrap)


def label(task, step, y, logprob=.8, first=True):
    return {"task_id": task, "step_id": step, "revision": {"y": y, "reason_code": "fixture"},
            "first_semantic_attempt": first, "original_step": {"raw_logprobs": {
                "available": True, "degenerate": False, "geo_mean_prob": logprob}}}


class NewSignalEvaluationTests(unittest.TestCase):
    def test_error_type_descriptions_count_missing_and_preserve_zero_scores(self):
        rows = [{"y": 1, "reason_code": "correct_a", "self_report": 1., "consistency": 0.},
                {"y": 1, "reason_code": "correct_b", "self_report": None, "consistency": None},
                {"y": 0, "reason_code": "fabricated", "self_report": .2, "consistency": None},
                {"y": 0, "reason_code": "fabricated", "self_report": .4, "consistency": 1.},
                {"y": 0, "reason_code": "wrong_unit", "self_report": None, "consistency": 0.}]
        before = copy.deepcopy(rows)
        stats = {r["group"]: r for r in error_type_diagnostics(rows)}
        self.assertEqual(sum(r["n"] for r in stats.values()), 5)
        self.assertEqual(stats["correct"]["n"], 2)
        self.assertEqual(stats["correct"]["signals"]["consistency"]["mean"], 0.)
        self.assertEqual(stats["fabricated"]["signals"]["consistency"]["n_scored"], 1)
        self.assertAlmostEqual(stats["fabricated"]["signals"]["self_report"]["mean"], .3)
        self.assertIsNone(stats["wrong_unit"]["signals"]["self_report"]["mean"])
        self.assertEqual(rows, before)

    def test_missing_not_imputed_and_genuine_zero_remains_scored(self):
        calls = [label("a", 1, 0), label("b", 1, 1), label("c", 1, 0)]
        scores = [{"task_id": "a", "step_id": 1, "self_report": 0., "consistency": 0., "consistency_k": 5},
                  {"task_id": "b", "step_id": 1, "self_report": None, "consistency": 1., "consistency_k": 5}]
        rows, audit = join_scores(calls, scores)
        self.assertEqual(rows[0]["self_report"], 0.)
        self.assertEqual(rows[0]["consistency"], 0.)
        self.assertIsNone(rows[1]["self_report"])
        self.assertIsNone(rows[2]["consistency"])
        self.assertEqual(audit["missing_score_rows"], 1)
        co = evaluate_rows(rows, 10)["all_calls"]
        self.assertEqual(co["own_coverage"]["self_report"]["n"], 1)
        self.assertIsNone(co["own_coverage"]["self_report"]["signals"]["self_report"]["AUROC"])
        self.assertEqual(co["own_coverage"]["consistency"]["n"], 2)
        self.assertEqual(co["common_all"]["n"], 1)

    def test_invalid_scores_never_clipped_or_silently_converted(self):
        calls = [label("a", 1, 0), label("b", 1, 1)]
        rows, audit = join_scores(calls, [
            {"task_id": "a", "step_id": 1, "self_report": True, "consistency": .8},
            {"task_id": "b", "step_id": 1, "self_report": float("nan"), "consistency": 2., "consistency_k": 5}])
        self.assertTrue(all(r["self_report"] is None for r in rows))
        self.assertTrue(all(r["consistency"] is None for r in rows))
        self.assertEqual(audit["invalid_scores"]["self_report"], 2)
        self.assertEqual(audit["invalid_scores"]["consistency_missing_or_invalid_k"], 1)

    def test_duplicate_unknown_and_noninteger_keys_are_rejected(self):
        calls = [label("a", 1, 0)]
        row = {"task_id": "a", "step_id": 1, "self_report": .5}
        with self.assertRaisesRegex(ValueError, "Duplicate score"):
            join_scores(calls, [row, row])
        with self.assertRaisesRegex(ValueError, "no frozen"):
            join_scores(calls, [{**row, "task_id": "absent"}])
        with self.assertRaisesRegex(ValueError, "integer step_id"):
            join_scores(calls, [{**row, "step_id": "1"}])

    def test_task_cluster_bootstrap_keeps_calls_together_and_pairs_deltas(self):
        # Each task contains both classes. A row bootstrap can select one class;
        # a task bootstrap cannot. Identical paired scores give exactly zero deltas.
        y = np.array([0, 1, 0, 1])
        p = np.array([.2, .8, .1, .9])
        boot = paired_bootstrap({"logprob": p, "self_report": p}, y, ["a", "a", "b", "b"], 100, 5)
        self.assertEqual(boot["AUROC"]["logprob"]["valid_resamples"], 100)
        self.assertEqual(boot["AUROC_difference_vs_logprob"]["self_report"]["ci95"], [0., 0.])
        self.assertEqual(boot, paired_bootstrap({"logprob": p, "self_report": p}, y, ["a", "a", "b", "b"], 100, 5))

    def test_tied_consistency_captures_random_expectation_not_file_order(self):
        self.assertAlmostEqual(error_capture(np.array([.5]*5), np.array([0, 0, 1, 1, 1])), .2)
        self.assertEqual(metrics([.5]*5, [0, 0, 1, 1, 1])["AUROC"], .5)

    def test_pair_comparisons_use_the_same_rows_and_frozen_dedup(self):
        calls = [label("a", 1, 0, .99), label("a", 2, 0, .98, False),
                 label("b", 1, 1, .7), label("c", 1, 1, .6)]
        scores = [{"task_id": "a", "step_id": 1, "self_report": .1, "consistency": .2, "consistency_k": 5},
                  {"task_id": "a", "step_id": 2, "self_report": .1, "consistency": .2, "consistency_k": 5},
                  {"task_id": "b", "step_id": 1, "self_report": .9, "consistency": .8, "consistency_k": 5}]
        original = copy.deepcopy(calls)
        rows, _ = join_scores(calls, scores)
        result = evaluate_rows(rows, 20)
        self.assertEqual(calls, original)
        own = result["all_calls"]["own_coverage"]
        self.assertEqual(own["logprob"]["n"], 4)
        self.assertEqual(own["self_report"]["n"], 3)
        pair = result["all_calls"]["paired_vs_logprob"]["self_report"]
        self.assertEqual(pair["signals"]["logprob"]["n"], pair["signals"]["self_report"]["n"])
        self.assertEqual(pair["paired_AUROC_difference_vs_logprob"]["self_report"]["point"], 1.)
        self.assertEqual(result["first_semantic_attempt"]["population_n"], 3)
        self.assertEqual(result["first_semantic_attempt"]["common_all"]["n"], 2)

    def test_real_frozen_calls_join_without_changing_old_labels_or_logprob(self):
        path = ROOT / "outputs/legacy_binary_v2/calls_binary.jsonl"
        before = path.read_bytes()
        calls = [json.loads(s) for s in before.decode().splitlines()]
        rows, _ = join_scores(calls, [])
        stat = metrics([r["logprob"] for r in rows], [r["y"] for r in rows])
        self.assertEqual((stat["n"], stat["correct"], stat["errors"]), (198, 158, 40))
        self.assertAlmostEqual(stat["AUROC"], .635126582278481)
        self.assertEqual(sum(r["first_semantic_attempt"] for r in rows), 130)
        self.assertEqual(sum(r["before_first_mock_error"] for r in rows), 100)
        self.assertEqual(path.read_bytes(), before)

    def test_history_subset_reuses_frozen_flag_and_validates_required_presence(self):
        calls = [label("a", 1, 0), label("a", 2, 1), label("b", 1, 1)]
        calls[0]["before_first_mock_error"] = True
        calls[1]["before_first_mock_error"] = False
        calls[2]["before_first_mock_error"] = True
        rows, audit = join_scores(calls, [], require_history_flags=True)
        co = evaluate_rows(rows, 10)["before_first_mock_error"]
        self.assertEqual(co["population_n"], 2)
        self.assertEqual(co["own_coverage"]["logprob"]["signals"]["logprob"]["errors"], 1)
        self.assertEqual(audit["missing_history_flags"], 0)
        del calls[0]["before_first_mock_error"]
        with self.assertRaisesRegex(ValueError, "Missing frozen history"):
            join_scores(calls, [], require_history_flags=True)
        # Lightweight synthetic inputs remain compatible, with an explicit audit.
        self.assertEqual(join_scores(calls, [])[1]["missing_history_flags"], 1)

    def test_public_summary_counts_are_allowlisted_and_do_not_read_runtime_logs(self):
        scores = [{"consistency_k": 3, "n_samples": 2, "n_agree": 1,
                   "n_strict_agree": 0, "n_no_tool": 1, "n_truncated": 1},
                  {"consistency_k": 3, "n_samples": 3, "n_agree": 3,
                   "n_strict_agree": 1, "n_no_tool": 0, "n_truncated": 0}]
        summary = {"successful_requests": 7, "failed_requests": 0,
                   "samples": 5, "no_tool_samples": 1, "multiple_tool_samples": 2,
                   "truncated_samples": 1, "prompt_tokens": 20,
                   "completion_tokens": 10, "total_tokens": 30, "usage_records": 7,
                   "successful_responses_without_usage": 0,
                   "private_extra": "must not propagate"}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out/"collection_summary.json").write_text(json.dumps(summary))
            # A runtime request file must be ignored, even if present and malformed.
            (out/"requests.jsonl").write_text("not a public research artifact")
            (out/"protocol.json").write_text(json.dumps({"consistency_thinking": False, "consistency_k": 3}))
            stats = collection_diagnostics(scores, out)
        self.assertEqual(stats["quality"]["totals"]["n_samples"], 5)
        self.assertEqual(stats["quality"]["missing_samples_with_known_count"], 1)
        self.assertEqual(stats["quality"]["reported_sample_counts"]["multiple_tool_samples"], 2)
        self.assertEqual((stats["cost"]["successful_requests"], stats["cost"]["failed_requests"]), (7, 0))
        self.assertEqual((stats["cost"]["prompt_tokens"], stats["cost"]["completion_tokens"],
                          stats["cost"]["total_tokens"]), (20, 10, 30))
        self.assertNotIn("private_extra", json.dumps(stats))
        self.assertNotIn("must not propagate", json.dumps(stats))
        self.assertIs(stats["protocol"]["consistency_thinking"], False)

    def test_missing_or_invalid_public_counts_remain_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            stats = collection_diagnostics([], out)
            self.assertIs(stats["summary"]["file_present"], False)
            self.assertIsNone(stats["cost"]["successful_requests"])
            self.assertIsNone(stats["cost"]["total_tokens"])
            (out/"collection_summary.json").write_text(json.dumps({
                "successful_requests": True, "failed_requests": -1, "total_tokens": "123",
                "truncated_samples": 0}))
            stats = collection_diagnostics([], out)
            self.assertIsNone(stats["cost"]["successful_requests"])
            self.assertIsNone(stats["cost"]["failed_requests"])
            self.assertIsNone(stats["cost"]["total_tokens"])
            self.assertEqual(stats["quality"]["reported_sample_counts"]["truncated_samples"], 0)
            (out/"collection_summary.json").write_text("[1,2]")
            stats = collection_diagnostics([], out)
            self.assertIs(stats["summary"]["parse_error"], True)
            self.assertIsNone(stats["cost"]["successful_requests"])

    def test_secondary_strict_score_does_not_change_primary_common_cases(self):
        calls = [label("a", 1, 0), label("b", 1, 1)]
        scores = [{"task_id": "a", "step_id": 1, "self_report": .1,
                   "consistency": 0., "strict_consistency": 0., "consistency_k": 3},
                  {"task_id": "b", "step_id": 1, "self_report": .9,
                   "consistency": 1., "strict_consistency": None, "consistency_k": 3}]
        rows, _ = join_scores(calls, scores)
        cohort = evaluate_rows(rows, 10)["all_calls"]
        self.assertEqual(cohort["common_all"]["n"], 2)
        strict = cohort["secondary_strict_consistency"]
        self.assertEqual(strict["n"], 1)
        self.assertIsNone(strict["signals"]["strict_consistency"]["AUROC"])

    def test_end_to_end_artifacts_preserve_inputs_and_report_partial_coverage(self):
        calls = [label("a", 1, 0), label("a", 2, 1), label("b", 1, 1)]
        scores = [{"task_id": "a", "step_id": 1, "self_report": .1, "consistency": 0., "consistency_k": 3},
                  {"task_id": "a", "step_id": 2, "self_report": .9, "consistency": 1., "consistency_k": 3}]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            labels_path, scores_path = out/"labels.jsonl", out/"scores.jsonl"
            labels_path.write_text("\n".join(json.dumps(r) for r in calls)+"\n")
            scores_path.write_text("\n".join(json.dumps(r) for r in scores)+"\n")
            (out/"protocol.json").write_text(json.dumps({"consistency_thinking": False, "consistency_k": 3}))
            (out/"collection_summary.json").write_text(json.dumps({"successful_requests": 792}))
            # Recomputing to another directory retains the input score provenance.
            # Output-local metadata must not override the actual collection source.
            (out/"report").mkdir()
            (out/"report/protocol.json").write_text(json.dumps({"consistency_thinking": True, "consistency_k": 99}))
            (out/"report/collection_summary.json").write_text(json.dumps({"successful_requests": 1}))
            before = labels_path.read_bytes(), scores_path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate(labels_path, scores_path, out/"report", 10)
            self.assertEqual(before, (labels_path.read_bytes(), scores_path.read_bytes()))
            self.assertEqual(result["input_audit"]["missing_score_rows"], 1)
            self.assertEqual(result["cohorts"]["all_calls"]["common_all"]["n"], 2)
            diagnostics = result["collection_diagnostics"]
            self.assertIs(diagnostics["protocol"]["consistency_thinking"], False)
            self.assertEqual(diagnostics["protocol"]["consistency_k"], 3)
            self.assertEqual(diagnostics["cost"]["successful_requests"], 792)
            self.assertEqual(json.loads((out/"report/new_signal_metrics.json").read_text())["evaluation_model_requests"], 0)
            self.assertGreater((out/"report/new_signal_comparison.png").stat().st_size, 1000)
            report = (out/"report/新信号评估.md").read_text()
            self.assertIn("2/3", report)
            self.assertIn(f"--scores {result['paths']['scores']}", report)
            self.assertIn(f"--out-dir {result['paths']['out_dir']}", report)
            self.assertIn("仅做 CPU 离线统计", report)
            self.assertIn("--no-consistency-thinking --cooldown 1", report)
            for name, path in result["paths"].items():
                self.assertFalse(Path(path).is_absolute(), name)
                self.assertNotIn(str(ROOT), path)
            self.assertNotIn(str(ROOT), report)
            self.assertEqual((ROOT/result["paths"]["scores"]).resolve(), scores_path.resolve())


if __name__ == "__main__":
    unittest.main()
