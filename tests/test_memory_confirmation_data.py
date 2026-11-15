"""Checks for fixed confirmation pairs and independently typed lures."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from memory_confirmation_data import CONFIRMATION_COUNTS, load_confirmation_cases
from memory_experiment_data import evaluate_action


class ConfirmationCasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_confirmation_cases()

    def test_matched_variants_isolate_the_added_evidence(self):
        pairs = zip(self.cases[::2], self.cases[1::2])
        missing_categories = Counter()
        for missing, complete in pairs:
            with self.subTest(base=missing["base_task_id"]):
                self.assertEqual(missing["base_task_id"], complete["base_task_id"])
                self.assertEqual(missing["history"], complete["history"])
                self.assertEqual(missing["api_documents"], complete["api_documents"])
                self.assertTrue(complete["requirement"].startswith(missing["requirement"] + " "))
                self.assertEqual(missing["expected"]["kind"], "clarify")
                self.assertEqual(complete["expected"]["kind"], "call")
                field, = missing["expected"]["fields"]
                self.assertEqual(missing["parameter_sources"][field], {"source": "missing"})
                self.assertEqual(complete["parameter_sources"][field]["source"], "requirement")
                known = set(missing["parameter_sources"]) - {field}
                for name in known:
                    self.assertEqual(missing["parameter_sources"][name], complete["parameter_sources"][name])
                missing_categories[missing["provenance"]["challenge_type"]] += 1
        self.assertEqual(missing_categories, CONFIRMATION_COUNTS)

    def test_same_distractor_is_unfounded_in_both_variants(self):
        for missing, complete in zip(self.cases[::2], self.cases[1::2]):
            field, = missing["expected"]["fields"]
            lure = {"kind": "call", "api_name": missing["target_tool"], "arguments": {
                **complete["expected"]["arguments"], **missing["provenance"]["tempting_args"],
            }}
            for case in (missing, complete):
                with self.subTest(task=case["task_id"]):
                    evaluation = evaluate_action(case, lure)
                    self.assertTrue(evaluation["fabricated"])
                    self.assertFalse(evaluation["correct"])
                    self.assertEqual(evaluation["unsupported_fields"], [field])

    def test_preexisting_tool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tool names overlap"):
            load_confirmation_cases(existing_cases=[self.cases[0]])
