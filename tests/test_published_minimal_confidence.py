"""Regression checks for the portable public confidence delivery."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from confidence_runtime import calibrate_consistency, digest, equivalent
from evaluate_minimal_validation import evaluate_rows, join_validation, run_diagnostics

OUT = ROOT / "outputs/confidence_minimal_v1"


def load(name):
    return json.loads((OUT / name).read_text())


def rows(name):
    return [json.loads(line) for line in (OUT / name).read_text().splitlines() if line.strip()]


class PublishedMinimalConfidenceTests(unittest.TestCase):
    def test_public_manifest_and_generated_provenance_match_actual_files(self):
        manifest = load("publication.json")
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, expected in manifest["public_artifact_sha256"].items():
            self.assertEqual(sha(OUT / name), expected, name)
        for name, expected in manifest["public_code_sha256"].items():
            self.assertEqual(sha(ROOT / "code" / name), expected, name)
        model = load("calibrator.json")
        self.assertEqual(model["provenance"]["code_sha256"], sha(ROOT / "code/calibrate_consistency.py"))
        self.assertEqual(model["provenance"]["collection_protocol_sha256"], sha(ROOT / "outputs/new_signals_fast_v1/protocol.json"))
        for field, expected in load("validation_metrics.json")["provenance"].items():
            path = ROOT / "code/evaluate_minimal_validation.py" if field == "evaluator_sha256" else OUT / field.removesuffix("_sha256")
            self.assertEqual(sha(path), expected, field)

    def test_compact_actions_reproduce_all_three_sample_scores(self):
        steps = rows("validation_steps.jsonl")
        self.assertEqual(len(steps), 50)
        all_seeds, no_tools = [], 0
        for step in steps:
            confidence = step["confidence"]
            samples = confidence["samples"]
            self.assertEqual(len(samples), 3)
            self.assertEqual([sample["index"] for sample in samples], [0, 1, 2])
            agreement = sum(equivalent(step["action"], sample["action"], step["api_documents"]) for sample in samples)
            self.assertEqual(confidence["raw_consistency"], agreement / 3)
            expected = calibrate_consistency(agreement / 3, load("calibrator.json"))
            self.assertEqual(confidence["p_correct"], expected["p_correct"])
            self.assertEqual(confidence["needs_verification"], expected["needs_verification"])
            for sample in samples:
                self.assertEqual(sample["seed"], 20261019 + int(digest(f"{step['task_id']}:{step['step_id']}:sample_{sample['index']}")[:7], 16))
                self.assertNotIn("response", sample)
                self.assertNotIn("request", sample)
                all_seeds.append(sample["seed"])
                no_tools += sample["action"] is None
            for message in step["messages_before"]:
                if message.get("tool_call_id"):
                    self.assertRegex(message["tool_call_id"], r"^call_\d{3}$")
        self.assertEqual(len(set(all_seeds)), 150)
        self.assertEqual(no_tools, 10)

    def test_reviewed_call_labels_and_task_outcomes_remain_distinct(self):
        model = load("calibrator.json")
        joined, quality = join_validation(rows("validation_steps.jsonl"), rows("validation_labels.jsonl"), model)
        metrics = evaluate_rows(joined, quality)
        run = run_diagnostics(OUT, metrics, joined)
        self.assertEqual((metrics["positive"], metrics["negative"]), (49, 1))
        self.assertEqual((run["executed_tasks"], run["zero_call_tasks"]), (30, 4))
        self.assertEqual((run["task_outcomes"]["successful_tasks"], run["task_outcomes"]["failed_tasks"]), (22, 8))
        self.assertTrue(metrics["limitations"]["calibrated_brier_exceeds_training_prior"])
        original = rows("validation_labels_original.jsonl")
        self.assertEqual(sum(row["y"] == 0 for row in original), 2)
        self.assertFalse(load("validation_label_review.json")["read_confidence_or_scores"])
        self.assertEqual(load("validation_label_review.json")["revised_counts"], {"correct": 49, "errors": 1})

    def test_cost_summary_is_sufficient_without_raw_requests(self):
        self.assertFalse((OUT / "validation_requests.jsonl").exists())
        joined, quality = join_validation(rows("validation_steps.jsonl"), rows("validation_labels.jsonl"), load("calibrator.json"))
        run = run_diagnostics(OUT, evaluate_rows(joined, quality), joined)
        cost = run["model_cost"]
        self.assertEqual(cost["source"], "collection_summary.json")
        self.assertEqual(cost["n_model_requests"], 230)
        self.assertEqual(cost["by_kind"]["primary"]["requests"], 80)
        self.assertEqual(cost["by_kind"]["confidence_probe"]["requests"], 150)
        self.assertEqual(cost["total_tokens_with_usage"], 182252)
        self.assertEqual(sum(row["confidence"]["cost"]["usage"]["total_tokens"] for row in rows("validation_steps.jsonl")), 92383)
        summary = load("collection_summary.json")
        summary["by_kind"]["primary"]["requests"] = True
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "collection_summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                run_diagnostics(folder, evaluate_rows(joined, quality), joined)

    def test_frozen_oof_and_validation_metrics_match_recorded_research(self):
        cal = load("calibration_metrics.json")
        self.assertAlmostEqual(cal["metrics"]["oof_calibrated"]["Brier"], .09174967393416061)
        self.assertEqual(len(rows("oof_predictions.jsonl")), 198)
        for fold in cal["folds"]:
            self.assertFalse(set(fold["train_tasks"]) & set(fold["valid_tasks"]))
        stats = load("validation_metrics.json")["cohorts"]["matched_all_signals"]["signals"]
        self.assertAlmostEqual(stats["p_correct"]["Brier"], .059879016472621796)
        self.assertAlmostEqual(stats["training_prior"]["Brier"], .05273135394347514)

    def test_export_contains_no_host_or_runtime_response_identifiers(self):
        forbidden = re.compile(r"/home(?:\d+)?/|/Users/|GPU-[0-9a-fA-F-]+|chatcmpl[-_]|0x[0-9a-fA-F]{8,}")
        for path in OUT.iterdir():
            if path.suffix in (".json", ".jsonl", ".md", ".csv"):
                self.assertIsNone(forbidden.search(path.read_text(encoding="utf-8-sig")), path.name)
        for name in ("validation_steps.jsonl", "validation_trajectories.jsonl"):
            for row in rows(name):
                self.assertNotIn("primary_response", row)
                self.assertNotIn("final_response", row)
                self.assertNotIn("endpoint_index", row)


if __name__ == "__main__":
    unittest.main()
