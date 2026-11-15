"""Check independent dual-source tasks and partial-memory controls."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from memory_experiment_data import evaluate_action
from memory_multislot_data import load_multislot_cases


class MultislotDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_multislot_cases()

    def test_each_holdout_has_two_distinct_prior_authorizations(self):
        for row in self.rows["test"]:
            with self.subTest(case=row["task_id"]):
                self.assertEqual(row["history"], [])
                records = {event["record_id"]: event for event in row["memory_events"]}
                fields = list(row["memory_fields"].items())
                self.assertEqual(len(fields), 2)
                self.assertEqual(len(row["permitted_memory_ids"]), 2)
                for (field, slot), record_id in zip(fields, row["permitted_memory_ids"]):
                    record = records[record_id]
                    self.assertEqual(record["slot"], slot)
                    self.assertEqual(record["profile_id"], row["profile_id"])
                    self.assertEqual(row["expected"]["arguments"][field], record["value"])
                    self.assertEqual(row["parameter_sources"][field]["source"], "history")
                self.assertTrue(evaluate_action(row, row["expected"])["correct"])

    def test_partial_records_require_targeted_questions(self):
        partial = [row for row in self.rows["validation"] if row["condition"] != "current_update"]
        self.assertEqual(len(partial), 6)
        for row in partial:
            with self.subTest(case=row["task_id"]):
                (unknown,) = row["expected"]["fields"]
                (known,) = row["memory_fields"].keys() - {unknown}
                self.assertEqual(len(row["permitted_memory_ids"]), 1)
                self.assertEqual(row["parameter_sources"][known]["source"], "history")
                precise = {"kind": "clarify", "text": f"Please provide {unknown.replace('_', ' ')}."}
                redundant = {"kind": "clarify", "text": precise["text"] + f" Please provide {known.replace('_', ' ')}."}
                self.assertTrue(evaluate_action(row, precise)["correct"])
                self.assertFalse(evaluate_action(row, redundant)["correct"])

    def test_current_override_retains_the_other_saved_record(self):
        for row in self.rows["validation"]:
            if row["condition"] != "current_update":
                continue
            with self.subTest(case=row["task_id"]):
                fields = row["memory_fields"]
                (current,) = [f for f in fields if row["parameter_sources"][f]["source"] == "requirement"]
                (other,) = fields.keys() - {current}
                self.assertEqual(row["parameter_sources"][other]["source"], "history")
                earlier = next(event for event in row["memory_events"] if event["slot"] == fields[current])
                self.assertIn(earlier["record_id"], row["permitted_memory_ids"])
                stale = {**row["expected"], "arguments": {**row["expected"]["arguments"], current: earlier["value"]}}
                self.assertEqual(evaluate_action(row, stale)["unsupported_fields"], [current])
