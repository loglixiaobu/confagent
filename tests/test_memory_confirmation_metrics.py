"""Checks for the directional paired test and the no-abstention guard."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from report_memory_confirmation import analyze_confirmation


def decisions(benefits=6, regressions=0, repaired_action="clarify"):
    rows = []
    for index in range(24):
        for expected in ("clarify", "call"):
            for arm in ("A", "B"):
                kind = expected
                if expected == "clarify" and index < benefits:
                    kind = "call" if arm == "A" else repaired_action
                elif expected == "clarify" and index < benefits + regressions and arm == "B":
                    kind = "call"
                fabricated = expected == "clarify" and kind == "call"
                rows.append({
                    "task_id": f"{expected}-{index}", "arm": arm, "expected": {"kind": expected},
                    "action": {"kind": kind}, "finish_reason": "length" if kind == "other" else "stop",
                    "memory_rule_ids": ["source-check"] if arm == "B" else [],
                    "evaluation": {"correct": kind == expected, "fabricated": fabricated,
                                   "unsupported_fields": ["id"] if fabricated else [],
                                   "reasonable_clarification": kind == expected == "clarify",
                                   "unnecessary_clarification": kind == "clarify" and expected == "call"},
                    "cost": {"requests": 1, "total_tokens": 10, "prompt_tokens": 8, "completion_tokens": 2},
                })
    return rows


class ConfirmationMetricsTests(unittest.TestCase):
    def test_six_paired_repairs_are_confirmed(self):
        summary = analyze_confirmation(decisions())
        self.assertEqual(summary["p_value"], 1 / 64)
        self.assertEqual(summary["absolute_reduction"], 6 / 24)
        self.assertTrue(summary["confirmed_positive"])

    def test_disagreement_direction_changes_the_test(self):
        summary = analyze_confirmation(decisions(regressions=1))
        self.assertEqual(summary["p_value"], 8 / 128)
        self.assertFalse(summary["confirmed_positive"])

    def test_refusing_to_act_cannot_show_a_benefit(self):
        summary = analyze_confirmation(decisions(repaired_action="other"))
        self.assertEqual(summary["p_value"], 1 / 64)
        self.assertFalse(summary["guardrails_passed"]["resolved_fabrications"])
        self.assertFalse(summary["guardrails_passed"]["no_decision_or_truncation"])
        self.assertFalse(summary["confirmed_positive"])
