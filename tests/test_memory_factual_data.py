"""Regression checks for stored-fact controls and current-input isolation."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from memory_experiment_data import evaluate_action
from memory_factual_data import load_factual_cases, matching_factual_record


class FactualMemoryDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_factual_cases()

    def test_saved_facts_stay_out_of_the_current_conversation(self):
        for row in self.cases["test"]:
            with self.subTest(task=row["task_id"]):
                record = matching_factual_record(row)
                self.assertIsNotNone(record)
                self.assertEqual(row["history"], [])
                self.assertEqual(row["expected"]["arguments"][row["memory_field"]], record["value"])
                value = record["value"]
                values = value if isinstance(value, list) else [value]
                for item in values:
                    self.assertIn(str(item), record["observation"][0]["content"])
                    self.assertNotIn(str(item).casefold(), row["requirement"].casefold())

    def test_missing_wrong_profile_and_wrong_slot_do_not_retrieve(self):
        for row in self.cases["validation"]:
            if row["condition"] != "current_update":
                with self.subTest(task=row["task_id"]):
                    self.assertIsNone(matching_factual_record(row))
                    self.assertEqual(row["expected"]["kind"], "clarify")
                    self.assertEqual(row["parameter_sources"][row["memory_field"]], {"source": "missing"})

    def test_explicit_current_value_wins_over_stale_confirmed_record(self):
        for row in self.cases["validation"]:
            if row["condition"] == "current_update":
                with self.subTest(task=row["task_id"]):
                    record = matching_factual_record(row)
                    self.assertIsNotNone(record)
                    field = row["memory_field"]
                    stale = {**row["expected"], "arguments": {
                        **row["expected"]["arguments"], field: record["value"],
                    }}
                    self.assertTrue(evaluate_action(row, row["expected"])["correct"])
                    self.assertEqual(evaluate_action(row, stale)["unsupported_fields"], [field])
