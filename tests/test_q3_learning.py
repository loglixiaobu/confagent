"""Critical source eligibility, public features and fitted-policy behavior."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from q3_three_layer.learning import ACTIONS, FEATURE_NAMES, RidgePolicy, build_memory, features, paired_gate

def experience(task_id="source-a", entity="station-a", value="731.25", factor="1000"):
    return {
        "task_id": task_id,
        "contract": {"source_value": value, "source_unit": "km", "target_unit": "m", "entity": entity},
        "trace": {
            "tool": "ConvertUnits", "source_value": value, "source_unit": "km", "target_unit": "m",
            "multiplier": factor, "result": str(float(value) * float(factor)),
            "executed": True, "success": True,
        },
        "certified": True,
        "origin": "tool_assisted_repair",
        "draft": {"value": value, "unit": "m"},
        "expected": "private-answer-that-must-not-be-copied",
    }

def pair():
    return [experience(), experience("source-b", "station-b", "942.5")]

def reward_rows(rewards):
    vector = features({"source_value": "2.5", "source_unit": "km", "target_unit": "m"},
                      {"value": "2.5", "unit": "m"}, True)
    return [{"features": vector, "action": action, "reward": rewards[action]}
            for _ in range(4) for action in ACTIONS], vector

class StructuredMemoryTests(unittest.TestCase):
    def test_memory_requires_actual_certified_traces_and_multiple_entities(self):
        self.assertEqual(build_memory([experience()]), [])
        same_entity = pair()
        same_entity[1]["contract"]["entity"] = same_entity[0]["contract"]["entity"]
        self.assertEqual(build_memory(same_entity), [])
        for field, replacement in (("trace", None), ("trace", {}), ("certified", None), ("certified", 1)):
            records = pair()
            records[1][field] = replacement
            self.assertEqual(build_memory(records), [])
        for key, replacement in (("executed", False), ("success", None), ("result", "0"),
                                 ("source_value", "8"), ("source_unit", "MiB")):
            records = pair()
            records[1]["trace"][key] = replacement
            self.assertEqual(build_memory(records), [])

class PublicFeatureTests(unittest.TestCase):
    def test_hidden_metadata_and_entity_changes_do_not_change_features(self):
        contract = {"source_value": "2.5", "source_unit": "km", "target_unit": "m", "entity": "a"}
        draft = {"value": "2.5", "unit": "m"}
        baseline = features(contract, draft, True)
        changed = dict(contract, entity="heldout-new", task_id="secret-id", split="test", family="qa",
                       expected="2500", correctness=True, gold="2500", reward=1.0)
        self.assertEqual(baseline, features(changed, dict(draft, expected=999, split="train"), True))
        self.assertEqual(len(baseline), len(FEATURE_NAMES))
        self.assertTrue(all(np.isfinite(baseline)))

class FittedPolicyTests(unittest.TestCase):
    def test_fitted_rewards_change_coefficients_and_decisions(self):
        rows, vector = reward_rows({"keep": 0.0, "memory": 0.3, "verify": 0.9})
        first = RidgePolicy.fit(rows)
        opposite_rows, _ = reward_rows({"keep": 0.9, "memory": 0.3, "verify": 0.0})
        second = RidgePolicy.fit(opposite_rows)
        self.assertEqual(first.choose(vector, ACTIONS), "verify")
        self.assertEqual(second.choose(vector, ACTIONS), "keep")
        self.assertFalse(np.array_equal(first.coefficients["keep"], second.coefficients["keep"]))
        self.assertEqual(first.counts, {action: 4 for action in ACTIONS})

    def test_action_masks_and_unobserved_actions_are_respected(self):
        rows, vector = reward_rows({"keep": 0.0, "memory": 0.3, "verify": 0.9})
        policy = RidgePolicy.fit(rows)
        self.assertEqual(policy.choose(vector, ["keep", "memory"]), "memory")
        self.assertEqual(policy.choose(vector, ["keep"]), "keep")
        sparse = RidgePolicy.fit([row for row in rows if row["action"] == "keep"])
        self.assertEqual(sparse.choose(vector, ACTIONS), "keep")
        with self.assertRaises(ValueError):
            sparse.choose(vector, ["verify"])
        with self.assertRaises(ValueError):
            policy.choose(vector, [])

class PairedGateTests(unittest.TestCase):
    def test_gate_distinguishes_fixes_harms_and_pair_coverage(self):
        result = paired_gate([False, False, True, True], [True, False, True, True])
        self.assertEqual((result["F"], result["H"], result["n"], result["pass"]), (1, 0, 4, True))
        harmful = paired_gate([False, False, True, True], [True, True, False, True])
        self.assertEqual((harmful["F"], harmful["H"], harmful["pass"]), (2, 1, False))
        self.assertEqual(paired_gate([False], [True])["reason"], "insufficient_pairs")
        neutral = paired_gate([True] * 4, [True] * 4)
        self.assertFalse(neutral["pass"])
        self.assertTrue(paired_gate([True] * 4, [True] * 4, require_fix=False)["pass"])

    def test_unknown_truthy_labels_and_missing_pairs_fail_closed(self):
        for unknown in (None, "correct", 1, np.bool_(True)):
            result = paired_gate([False, True, True, False], [True, True, True, unknown])
            self.assertFalse(result["pass"])
            self.assertEqual(result["reason"], "unknown_outcomes")
        self.assertEqual(paired_gate([False] * 4, [True] * 3)["reason"], "length_mismatch")
        self.assertFalse(paired_gate([], [])["pass"])
