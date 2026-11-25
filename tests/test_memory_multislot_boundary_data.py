"""Consistency and gold-isolation tests for the independent V9 extension."""
from collections import Counter
from pathlib import Path
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from memory_experiment_data import evaluate_action  # noqa: E402
from memory_multislot_boundary_data import (  # noqa: E402
    BOUNDARY_COUNTS,
    BOUNDARY_TEST_COUNTS,
    EXPECTED_CONDITIONS,
    EXPECTED_TEST_CONDITIONS,
    load_boundary_candidate_replay,
    load_boundary_model_inputs,
    load_multislot_boundary_cases,
)
from memory_multislot_data import load_multislot_cases  # noqa: E402


class MultislotBoundaryDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load_boundary_model_inputs()
        cls.cases = load_multislot_boundary_cases()
        cls.candidates = load_boundary_candidate_replay()

    def test_development_and_test_cover_four_categories_with_distinct_tools(self):
        self.assertEqual(len(self.cases), 20)
        self.assertEqual(len({case["target_tool"] for case in self.cases}), 20)
        by_split = {split: [case for case in self.cases if case["dataset_split"] == split]
                    for split in ("development", "test")}
        self.assertEqual({key: len(value) for key, value in by_split.items()},
                         {"development": 8, "test": 12})
        self.assertEqual(Counter(case["boundary_category"] for case in by_split["development"]),
                         BOUNDARY_COUNTS)
        self.assertEqual(Counter(case["boundary_category"] for case in by_split["test"]),
                         BOUNDARY_TEST_COUNTS)
        self.assertEqual(Counter(row["category"] for row in self.candidates
                                 if row["expected_label"] == "correct"), BOUNDARY_COUNTS)
        self.assertEqual(Counter(case["condition"] for case in by_split["development"]),
                         EXPECTED_CONDITIONS)
        self.assertEqual(Counter(case["condition"] for case in by_split["test"]),
                         EXPECTED_TEST_CONDITIONS)

    def test_only_development_has_handwritten_candidates(self):
        raw = json.loads((ROOT / "data" / "memory_multislot_boundary_cases.json").read_text(
            encoding="utf-8"))
        self.assertTrue(all(len(bundle["offline_candidates"]) >= 2
                            for bundle in raw["development"]))
        self.assertTrue(all("offline_candidates" not in bundle for bundle in raw["test"]))
        self.assertEqual(len(self.candidates), 16)
        self.assertEqual({row["split"] for row in self.candidates}, {"development"})

    def test_model_visible_inputs_contain_no_gold_or_forbidden_records(self):
        forbidden = {"answer_key", "expected", "parameter_sources", "offline_candidates"}
        for row in self.inputs:
            with self.subTest(task=row["task_id"]):
                self.assertFalse(forbidden & set(row))
                self.assertEqual(
                    {event["record_id"] for event in row["permitted_memory"]},
                    set(row["permitted_memory_ids"]),
                )
                serialized = json.dumps(row, ensure_ascii=False)
                self.assertNotIn("expected_label", serialized)
        by_task = {row["task_id"]: json.dumps(row, ensure_ascii=False) for row in self.inputs}
        self.assertNotIn("Loading Dock Access Update",
                         by_task["q2-boundary-email-revoked-subject-next"])
        self.assertNotIn("Dr. Maren Vale",
                         by_task["q2-boundary-provider-wrong-profile-next"])
        self.assertNotIn("2034-06-18",
                         by_task["q2-test-registration-revoked-date-next"])
        self.assertNotIn("68.0",
                         by_task["q2-test-clothing-foreign-humidity-next"])
        self.assertNotIn("2034-07-03 07:25:00",
                         by_task["q2-test-health-record-revoked-time-next"])

    def test_split_loaders_keep_test_inputs_candidate_free(self):
        self.assertEqual(len(load_boundary_model_inputs(split="development")), 8)
        self.assertEqual(len(load_boundary_model_inputs(split="test")), 12)
        self.assertEqual(len(load_multislot_boundary_cases(split="development")), 8)
        self.assertEqual(len(load_multislot_boundary_cases(split="test")), 12)

    def test_answer_keys_and_candidate_labels_match_existing_v9_scorer(self):
        by_task = {case["task_id"]: case for case in self.cases}
        for row in self.candidates:
            with self.subTest(case=row["case_id"], candidate=row["candidate_id"]):
                evaluation = evaluate_action(by_task[row["task_id"]], row["action"])
                self.assertEqual(evaluation, row["evaluation"])
                self.assertEqual(evaluation["correct"], row["expected_label"] == "correct")

    def test_original_v9_file_and_frozen_counts_remain_unchanged(self):
        raw = json.loads((ROOT / "data" / "memory_multislot_cases.json").read_text(encoding="utf-8"))
        self.assertEqual(len(raw["validation"]), 8)
        self.assertEqual(len(raw["test"]), 24)
        # Full loading is possible when the separately distributed API-Bank
        # corpus is present. The immutable raw counts are always checked here.
        if (ROOT / "data" / "api-bank").exists():
            loaded = load_multislot_cases()
            self.assertEqual({key: len(value) for key, value in loaded.items()},
                             {"validation": 8, "test": 24})


if __name__ == "__main__":
    unittest.main()
