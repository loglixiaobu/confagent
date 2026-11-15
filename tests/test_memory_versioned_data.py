"""Regression checks for independent versioned-memory cases."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from memory_experiment_data import evaluate_action
from memory_versioned_data import load_versioned_cases


class VersionedMemoryDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_versioned_cases()

    def test_updated_facts_are_independent_of_record_storage_order(self):
        reversed_rows, later_wrong_scope = 0, 0
        for row in self.cases["updated"]:
            timeline = row["memory_events"]
            matching = [event for event in timeline if event["profile_id"] == row["profile_id"]
                        and event["slot"] == row["requested_slot"]]
            sequences = [event["sequence"] for event in timeline]
            reversed_rows += sequences != sorted(sequences)
            latest = max(matching, key=lambda event: event["sequence"])
            later_wrong_scope += max(sequences) > latest["sequence"]
            self.assertEqual(len(matching), 2)
            self.assertEqual(row["permitted_memory_ids"], [latest["record_id"]])
            self.assertEqual(row["expected"]["arguments"][row["memory_field"]], latest["value"])
        self.assertGreater(reversed_rows, len(self.cases["updated"]) // 2)
        self.assertGreater(later_wrong_scope, 0)

    def test_revoked_and_wrong_scope_controls_permit_no_record(self):
        for row in self.cases["validation"]:
            if row["condition"] != "current_update":
                with self.subTest(task=row["task_id"]):
                    self.assertEqual(row["permitted_memory_ids"], [])
                    self.assertEqual(row["expected"], {"kind": "clarify", "fields": [row["memory_field"]]})
                    self.assertEqual(row["history"], [])

    def test_earlier_session_values_are_not_in_current_prompts(self):
        for rows in self.cases.values():
            for row in rows:
                self.assertEqual(row["history"], [])
                for event in row["memory_events"]:
                    if event["kind"] != "save":
                        self.assertNotIn("value", event)
                        continue
                    value = event["value"]
                    values = value if isinstance(value, list) else [value]
                    for item in values:
                        self.assertIn(str(item), event["observation"][0]["content"])
                        self.assertNotIn(str(item).casefold(), row["requirement"].casefold())

    def test_current_values_override_permitted_old_records(self):
        for row in self.cases["validation"]:
            if row["condition"] == "current_update":
                with self.subTest(task=row["task_id"]):
                    field = row["memory_field"]
                    old, = row["memory_events"]
                    self.assertEqual(row["permitted_memory_ids"], [old["record_id"]])
                    stale = {**row["expected"], "arguments": {**row["expected"]["arguments"], field: old["value"]}}
                    self.assertTrue(evaluate_action(row, row["expected"])["correct"])
                    self.assertEqual(evaluate_action(row, stale)["unsupported_fields"], [field])
