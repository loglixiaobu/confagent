"""Regression checks for per-field evidence and unsupported cross-slot reuse."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from run_memory_multislot import presented_case, used_disallowed_fact


class MultislotRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "memory_fields": {"city": "appointment_city", "name": "provider_name"},
            "parameter_sources": {
                "city": {"source": "requirement", "value": "Bath"},
                "name": {"source": "history", "value": "Pat"},
            },
            "expected": {"kind": "call", "api_name": "Schedule", "arguments": {"city": "Bath", "name": "Pat"}},
            "api_documents": [{
                "name": "Schedule",
                "input_parameters": {"city": {"type": "str"}, "name": {"type": "str"}},
            }],
            "target_tool": "Schedule",
            "memory_events": [{"kind": "save", "value": "York"}, {"kind": "save", "value": "Pat"}],
        }

    def test_current_value_remains_visible_without_its_older_recalled_fact(self):
        row = presented_case(self.case, [])
        self.assertEqual(row["expected"], {"kind": "clarify", "fields": ["name"]})
        self.assertEqual(row["parameter_sources"]["city"]["value"], "Bath")
        self.assertEqual(presented_case(self.case, ["name"])["expected"], self.case["expected"])

    def test_using_a_saved_value_for_another_field_is_disallowed(self):
        wrong = {"kind": "call", "api_name": "Schedule", "arguments": {"city": "Pat", "name": "Pat"}}
        self.assertTrue(used_disallowed_fact(self.case, wrong))
        self.assertFalse(used_disallowed_fact(self.case, self.case["expected"]))
