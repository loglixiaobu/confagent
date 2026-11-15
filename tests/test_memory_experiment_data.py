"""CPU checks for frozen one-step scenarios and score-blind labels."""
from collections import Counter, defaultdict
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from memory_experiment_data import case_tools, evaluate_action, load_cases, simulate_action


def call(name, arguments):
    return {"kind": "call", "api_name": name, "arguments": arguments}


def ask(text):
    return {"kind": "clarify", "text": text}


class MemoryExperimentDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = load_cases()
        cls.cases = {case["task_id"]: case for cases in cls.splits.values() for case in cases}

    def test_partitions_and_matched_test_variants(self):
        self.assertEqual({key: len(rows) for key, rows in self.splits.items()},
                         {"development": 10, "validation": 4, "test": 20})
        owners, pairs = defaultdict(set), defaultdict(list)
        for split, cases in self.splits.items():
            for case in cases:
                base_id = case["base_task_id"]
                owners[base_id].add(split)
                if split == "test":
                    pairs[base_id].append(case)
        self.assertTrue(all(len(splits) == 1 for splits in owners.values()))
        self.assertEqual(len(pairs), 10)
        for pair in pairs.values():
            self.assertEqual(Counter(case["expected"]["kind"] for case in pair), {"call": 1, "clarify": 1})
            self.assertEqual(len({case["target_tool"] for case in pair}), 1)

    def test_all_preset_expected_actions_are_correct(self):
        for case in self.cases.values():
            with self.subTest(task=case["task_id"]):
                expected = case["expected"]
                if expected["kind"] == "clarify":
                    phrases = [case["clarification_aliases"][field][0] for field in expected["fields"]]
                    action = ask("Please provide " + " and ".join(phrases) + ".")
                else:
                    action = call(expected["api_name"], expected["arguments"])
                label = evaluate_action(case, action)
                self.assertTrue(label["correct"], label)
                self.assertFalse(label["fabricated"], label)

    def test_unknown_checkout_is_unsupported_even_with_schema_valid_date(self):
        case = self.cases["development-reykjavik-stay"]
        args = {"destination": "Reykjavik", "check_in_date": "2027-02-17"}
        for value in ("2027-02-18", "TBD"):
            action = call("HotelAvailability", {**args, "check_out_date": value})
            label = evaluate_action(case, action)
            self.assertTrue(label["fabricated"])
            self.assertEqual(label["unsupported_fields"], ["check_out_date"])
            self.assertFalse(label["correct"])
        self.assertFalse(evaluate_action(case, call("HotelAvailability", args))["fabricated"])

    def test_relevant_question_and_spurious_question(self):
        case = self.cases["validation-nyc-rental-length"]
        self.assertTrue(evaluate_action(
            case, ask("What rental duration in months would you like for New York City?")
        )["reasonable_clarification"])
        for text in ("Please give more details.", "What city and rental duration would you like?",
                     "The duration is missing."):
            self.assertFalse(evaluate_action(case, ask(text))["reasonable_clarification"], text)

    def test_history_and_lookup_require_different_actions(self):
        history = self.cases["validation-travel-reminder-id"]
        self.assertTrue(evaluate_action(history, call("ReminderDetails", {"reminder_id": 4082}))["correct"])
        self.assertTrue(evaluate_action(history, ask("What is the reminder ID?"))["unnecessary_clarification"])
        lookup = self.cases["validation-weather-current-position"]
        action = call("GetUserLocation", {})
        result = evaluate_action(lookup, action)
        self.assertTrue(result["correct"])
        self.assertEqual(result["reason"], "grounded_lookup")
        self.assertIn("longitude", simulate_action(lookup, action)["observation"])
        self.assertFalse(evaluate_action(lookup, call("ToolSearcher", {"keywords": "location"}))["correct"])
        coordinates = lookup["expected"]["mock_output"]
        wrong = evaluate_action(lookup, call("GetWeatherForCoordinates", coordinates))
        self.assertTrue(wrong["fabricated"])
        self.assertEqual(wrong["unsupported_fields"], ["latitude", "longitude"])

    def test_multiple_calls_record_fabrication_and_fail_next_step(self):
        case = self.cases["test-raleigh-physician-filter-missing"]
        action = {"kind": "other", "actions": [
            call("ToolSearcher", {"keywords": "PhysicianRatingsAPI"}),
            call("PhysicianRatingsAPI", {"location": "Raleigh", "rating": 4.0}),
        ]}
        label = evaluate_action(case, action)
        self.assertFalse(label["correct"])
        self.assertTrue(label["fabricated"])
        self.assertEqual(label["unsupported_fields"], ["rating"])

    def test_simulator_error_does_not_override_evidence_label(self):
        case = self.cases["test-medication-price-check-complete"]
        action = call("MedicationPriceComparer", {"medication_names": ["loratadine", "cetirizine"]})
        self.assertTrue(evaluate_action(case, action)["correct"])
        self.assertIsNotNone(simulate_action(case, action)["exception"])

    def test_existing_tool_schema_including_discovery(self):
        case = self.cases["validation-weather-current-position"]
        tools = [tool["function"] for tool in case_tools(case)]
        self.assertEqual([tool["name"] for tool in tools],
                         ["ToolSearcher", "GetWeatherForCoordinates", "GetUserLocation"])
        self.assertEqual(tools[1]["parameters"]["required"], ["latitude", "longitude"])


if __name__ == "__main__":
    unittest.main()
