"""Checks for the independently frozen unsupported-argument stress pairs."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from memory_experiment_data import DEFAULT_STRESS_CASES, evaluate_action, load_cases, load_stress_cases


class MemoryStressDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_cases()
        cls.stress = load_stress_cases(existing_cases=cls.original)

    def test_pairs_are_balanced_and_independent_of_original_partition(self):
        original_bases = {case["base_task_id"] for rows in self.original.values() for case in rows}
        pairs = defaultdict(list)
        for case in self.stress:
            pairs[case["base_task_id"]].append(case)
        self.assertFalse(original_bases.intersection(pairs))
        self.assertEqual(len(pairs), 10)
        for pair in pairs.values():
            self.assertEqual(Counter(case["expected"]["kind"] for case in pair), {"call": 1, "clarify": 1})

    def test_stress_lures_are_unsupported_and_matched_answers_are_valid(self):
        full_cases = {case["base_task_id"]: case for case in self.stress if case["expected"]["kind"] == "call"}
        for incomplete in (case for case in self.stress if case["expected"]["kind"] == "clarify"):
            with self.subTest(task=incomplete["task_id"]):
                full = full_cases[incomplete["base_task_id"]]
                missing_field, = incomplete["expected"]["fields"]
                gold_call = full["expected"]
                lure = incomplete["provenance"]["tempting_args"]
                guessed_call = {**gold_call, "arguments": {**gold_call["arguments"], **lure}}
                fabricated = evaluate_action(incomplete, guessed_call)
                self.assertTrue(fabricated["fabricated"])
                self.assertEqual(fabricated["unsupported_fields"], [missing_field])
                self.assertFalse(evaluate_action(full, guessed_call)["correct"])
                self.assertTrue(evaluate_action(full, gold_call)["correct"])
                alias = incomplete["clarification_aliases"][missing_field][0]
                question = {"kind": "clarify", "text": f"Please provide the {alias}."}
                self.assertTrue(evaluate_action(incomplete, question)["reasonable_clarification"])

    def test_loader_rejects_cross_partition_base_ids(self):
        raw = json.loads(DEFAULT_STRESS_CASES.read_text(encoding="utf-8"))
        raw["test"][0]["base_task_id"] = self.original["development"][0]["base_task_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlapping.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cross-partition"):
                load_stress_cases(path, existing_cases=self.original)


if __name__ == "__main__":
    unittest.main()
