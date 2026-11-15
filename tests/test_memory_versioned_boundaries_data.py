"""Validate scenario isolation and chronology in the extended boundary cohort."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from memory_experiment_data import evaluate_action
from memory_versioned_boundaries_data import (PRIOR_CASE_FILES, _prior_sets,
                                             load_boundaries_cases)


class VersionedBoundariesDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_boundaries_cases()

    def test_fresh_task_families_and_tools(self):
        self.assertEqual(len(PRIOR_CASE_FILES), 5)
        old_tasks, old_bases, _, old_tools = _prior_sets()
        cases = [row for split in self.cases.values() for row in split]
        tools = {row["target_tool"] for row in cases}
        self.assertEqual(len(cases), len(tools))
        self.assertFalse(tools & old_tools)
        self.assertFalse({row["task_id"] for row in cases} & old_tasks)
        self.assertFalse({row["base_task_id"] for row in cases} & old_bases)
        original_level1 = sum(row["provenance"]["tool_schema_references"][row["target_tool"]]["corpus"]
                              == "level1_response" for row in cases)
        self.assertEqual(original_level1, 22)

    def test_selected_version_ignores_earlier_values_and_out_of_scope_decoys(self):
        wrong_order = later_decoy = 0
        for row in self.cases["updated"]:
            with self.subTest(case=row["task_id"]):
                self.assertEqual(row["history"], [])
                events = row["memory_events"]
                timeline = sorted(
                    (event for event in events if event["profile_id"] == row["profile_id"]
                     and event["slot"] == row["requested_slot"]),
                    key=lambda event: event["sequence"],
                )
                earliest, latest = timeline
                result = row["expected"]
                field = row["memory_field"]
                self.assertEqual(row["permitted_memory_ids"], [latest["record_id"]])
                self.assertEqual(result["arguments"][field], latest["value"])
                stale = {**result, "arguments": {**result["arguments"], field: earliest["value"]}}
                self.assertEqual(evaluate_action(row, stale)["unsupported_fields"], [field])
                order = [event["sequence"] for event in events]
                wrong_order += order != sorted(order)
                later_decoy += max(order) > latest["sequence"]
        self.assertEqual(wrong_order, 15)
        self.assertEqual(later_decoy, 5)

    def test_controls_reveal_only_the_permitted_record(self):
        for row in self.cases["validation"]:
            if row["condition"] == "current_update":
                self.assertEqual(row["expected"]["kind"], "call")
                self.assertEqual(len(row["permitted_memory_ids"]), 1)
            else:
                self.assertEqual(row["expected"]["kind"], "clarify")
                self.assertEqual(row["permitted_memory_ids"], [])
