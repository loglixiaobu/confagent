"""CPU-only checks for leakage boundaries and fixed-action signal semantics."""
import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from collect_new_signals import (Guard, RETRY_PROMPT, collect, digest, equivalent, legacy_tool, parse_self,
                                 prepare, reconstruct, self_messages, validate_states)


def document(name="Lookup", parameters=None):
    return {
        "name": name,
        "description": "Find the requested record.",
        "input_parameters": parameters or {
            "id": {"type": "str", "description": "Record identifier."}},
        "output_parameters": {"value": {"type": "str", "description": "Recorded value."}},
    }


def fixtures():
    api = document()
    task = {
        "requirement": "Look up records 001, 002, and 003.",
        "response": "GOLD_FINAL_RESPONSE_SENTINEL",
        "apis": [
            {"api_name": "ToolSearcher", "input": {"keywords": "GOLD_SEARCH_INPUT_SENTINEL"},
             "output": {"output": api, "exception": None}},
            {"api_name": "Lookup", "input": {"id": "GOLD_ACTION_INPUT_SENTINEL"},
             "output": {"output": {"value": "GOLD_RUNTIME_OUTPUT_SENTINEL"}, "exception": None}},
        ],
    }
    trajectory = {
        "task_id": "l3-000", "final_success": False,
        "final_answer": "RECORDED_FINAL_SENTINEL",
        "config": {"stop_reason": "final_answer"},
        "steps": [
            {"step_id": i, "action_input": {"api_name": "Lookup", "arguments": {"id": f"00{i}"}},
             "observation": {"value": f"OBSERVATION_{i}_SENTINEL"},
             "matched_gold": False, "confidence": 0.937,
             "label_revision": {"y": 0, "reason": "LABEL_REASON_SENTINEL"}}
            for i in range(1, 4)
        ],
    }
    return task, trajectory


class ReconstructedStateTests(unittest.TestCase):
    def test_prefix_contains_only_earlier_observations_and_no_labels(self):
        task, trajectory = fixtures()
        states = reconstruct([trajectory], [task])
        self.assertEqual(len(states), 3)
        for index, state in enumerate(states):
            encoded = json.dumps(state)
            for observation_index in range(1, 4):
                marker = f"OBSERVATION_{observation_index}_SENTINEL"
                self.assertEqual(marker in encoded, observation_index <= index)
            for forbidden in ("matched_gold", "confidence", "label_revision", "LABEL_REASON_SENTINEL",
                              "RECORDED_FINAL_SENTINEL", "GOLD_FINAL_RESPONSE_SENTINEL",
                              "GOLD_ACTION_INPUT_SENTINEL", "GOLD_RUNTIME_OUTPUT_SENTINEL",
                              "GOLD_SEARCH_INPUT_SENTINEL"):
                self.assertNotIn(forbidden, encoded)
        self.assertEqual(len(states[0]["messages"]), 2)
        self.assertEqual(states[1]["messages"][-1]["role"], "tool")

    def test_later_observation_and_label_changes_cannot_change_earlier_states(self):
        task, trajectory = fixtures()
        original = reconstruct([trajectory], [task])
        changed = copy.deepcopy(trajectory)
        changed["steps"][1]["observation"] = {"value": "DIFFERENT_FUTURE_RESULT"}
        for step in changed["steps"]:
            step["matched_gold"] = True
            step["label_revision"] = {"y": 1, "reason": "DIFFERENT_LABEL"}
            step["confidence"] = 0.01
        revised = reconstruct([changed], [task])
        self.assertEqual(original[:2], revised[:2])
        self.assertNotEqual(original[2]["messages"], revised[2]["messages"])

    def test_visible_documents_are_metadata_and_all_original_tools_are_preunlocked(self):
        task, trajectory = fixtures()
        second = document("AnotherAPI")
        task["apis"].append({"api_name": "ToolSearcher", "input": {"keywords": "OTHER_GOLD_QUERY"},
                             "output": {"output": second}})
        first = reconstruct([trajectory], [task])[0]
        self.assertEqual([doc["name"] for doc in first["api_documents"]], ["Lookup", "AnotherAPI"])
        self.assertEqual([tool["function"]["name"] for tool in first["tools"]],
                         ["ToolSearcher", "Lookup", "AnotherAPI"])
        self.assertEqual(set(first["api_documents"][0]),
                         {"name", "description", "input_parameters", "output_parameters"})
        serialized = json.dumps(first["api_documents"])
        for marker in ("GOLD_ACTION_INPUT_SENTINEL", "GOLD_RUNTIME_OUTPUT_SENTINEL",
                       "GOLD_FINAL_RESPONSE_SENTINEL", "OTHER_GOLD_QUERY"):
            self.assertNotIn(marker, serialized)

    def test_legacy_list_of_strings_schema_stays_string(self):
        api = document("AddMeeting", {"attendees": {"type": "list(str)", "description": "Attendees."}})
        props = legacy_tool(api)["function"]["parameters"]["properties"]
        self.assertEqual(props["attendees"]["type"], "string")
        self.assertNotIn("items", props["attendees"])

    def test_retry_prompt_appears_after_second_identical_call(self):
        task, trajectory = fixtures()
        repeated = {"api_name": "Lookup", "arguments": {"id": "001"}}
        for step in trajectory["steps"]:
            step["action_input"] = copy.deepcopy(repeated)
        states = reconstruct([trajectory], [task])
        self.assertFalse(any(message.get("content") == RETRY_PROMPT for message in states[1]["messages"]))
        prompts = [message for message in states[2]["messages"] if message.get("content") == RETRY_PROMPT]
        self.assertEqual(len(prompts), 1)
        self.assertEqual(states[2]["messages"][-1], {"role": "user", "content": RETRY_PROMPT})
        self.assertEqual(states[2]["messages"][-2]["role"], "tool")

    def test_self_report_receives_docs_candidate_and_only_previous_results(self):
        task, trajectory = fixtures()
        state = reconstruct([trajectory], [task])[1]
        before = copy.deepcopy(state)
        messages = self_messages(state)
        encoded = json.dumps(messages)
        self.assertIn("Available API documentation", messages[0]["content"])
        self.assertIn("Record identifier", encoded)
        self.assertIn("I am about to execute: Lookup", messages[-2]["content"])
        self.assertIn('"id": "002"', messages[-2]["content"])
        self.assertIn("OBSERVATION_1_SENTINEL", encoded)
        for marker in ("OBSERVATION_2_SENTINEL", "OBSERVATION_3_SENTINEL", "LABEL_REASON_SENTINEL",
                       "GOLD_ACTION_INPUT_SENTINEL", "GOLD_RUNTIME_OUTPUT_SENTINEL"):
            self.assertNotIn(marker, encoded)
        self.assertEqual(state, before)


class PortablePreparationTests(unittest.TestCase):
    def test_prebuilt_states_roundtrip_and_frozen_protocol(self):
        task, trajectory = fixtures()
        states = reconstruct([trajectory], [task])
        serialized = "".join(json.dumps(state, ensure_ascii=False) + "\n" for state in states)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "states.jsonl"
            source.write_text(serialized)
            out = base / "probe"
            actual, protocol = prepare(out, 3, False, states_file=source)
            self.assertEqual(actual, states)
            self.assertEqual((out / "states.jsonl").read_text(), serialized)
            self.assertEqual(protocol["states_sha256"], digest(states))
            self.assertEqual(protocol["source_mode"], "prebuilt_states")
            self.assertIsNone(protocol["source_sha256"])
            self.assertIsNone(protocol["data_sha256"])
            self.assertNotIn(str(base), json.dumps(protocol))
            prepare(out, 3, False, states_file=source)
            with self.assertRaisesRegex(ValueError, "Frozen protocol changed"):
                prepare(out, 4, False, states_file=source)

    def test_prebuilt_state_schema_and_unique_keys(self):
        task, trajectory = fixtures()
        states = reconstruct([trajectory], [task])
        self.assertEqual(validate_states(states), states)
        for invalid in ([], [states[0], states[0]], [dict(states[0], candidate=None)],
                        [dict(states[0], messages={})], [dict(states[0], tools="wrong")],
                        [dict(states[0], api_documents=None)], [dict(states[0], step_id=True)],
                        [{key: value for key, value in states[0].items() if key != "candidate"}]):
            with self.assertRaises(ValueError):
                validate_states(invalid)

    def test_sequential_requests_are_checkpointed_and_resume_without_new_calls(self):
        task, trajectory = fixtures()
        state = reconstruct([trajectory], [task])[0]
        sampled = {"choices": [{"message": {"tool_calls": [{"function": {
            "name": "Lookup", "arguments": '{"id": "001"}'}}]}, "finish_reason": "tool_calls"}]}
        self_response = {"choices": [{"message": {"content": "0.8"}, "finish_reason": "stop"}]}
        responses = [self_response, sampled, sampled]
        create = Mock(side_effect=[SimpleNamespace(model_dump=lambda value=value: value)
                                   for value in responses])
        clients = []
        def make_client(**_):
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                                     close=Mock())
            clients.append(client)
            return client
        modules = {"openai": SimpleNamespace(OpenAI=make_client),
                   "httpx": SimpleNamespace(Client=lambda **_: object())}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            protocol = {"model": "Qwen3-8B", "consistency_k": 2, "consistency_thinking": False}
            with patch.dict(sys.modules, modules), patch("collect_new_signals.time.sleep") as sleep:
                collect(out, [state], protocol, 0, 1)
                first_scores = (out / "scores.jsonl").read_text()
                requests = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
                self.assertEqual(create.call_count, 3)
                self.assertEqual(sleep.call_count, 3)
                for record, call in zip(requests, create.call_args_list):
                    self.assertEqual(record["request"], call.kwargs)
                    self.assertEqual(record["payload_sha256"], digest(call.kwargs))
                    self.assertEqual(call.kwargs["seed"], 20261015 + int(digest(record["request_id"])[:7], 16))
                collect(out, [state], protocol, 0, 1)
                self.assertEqual(create.call_count, 3)
                self.assertEqual(sleep.call_count, 3)
                self.assertEqual((out / "scores.jsonl").read_text(), first_scores)
                score = json.loads(first_scores)
                self.assertEqual(score["self_report"], .8)
                self.assertEqual(score["consistency"], 1.0)
                self.assertEqual(score["n_samples"], 2)
        self.assertTrue(all(client.close.call_count == 1 for client in clients))

    def test_health_gate_is_optional_and_validates_health(self):
        Guard().wait()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "health.json"
            with self.assertRaises(RuntimeError):
                Guard(path).wait()
            fresh = {"status": "healthy", "can_submit": True, "updated_at": time.time()}
            path.write_text(json.dumps(fresh))
            Guard(path).wait()
            for changed in ({"status": "stopped"}, {"can_submit": "true"},
                            {"updated_at": time.time() - 60}, {"updated_at": True},
                            {"updated_at": time.time() + 60}):
                path.write_text(json.dumps(dict(fresh, **changed)))
                with self.assertRaises(RuntimeError):
                    Guard(path).wait()
            path.write_text(json.dumps(dict(fresh, can_submit=False)))
            def become_ready(_):
                path.write_text(json.dumps(fresh))
            with patch("collect_new_signals.time.sleep", side_effect=become_ready) as sleep:
                Guard(path).wait()
                sleep.assert_called_once_with(1)


class SemanticAgreementTests(unittest.TestCase):
    def agrees(self, name, left, right, parameter_types):
        api = document(name, {key: {"type": kind} for key, kind in parameter_types.items()})
        return equivalent({"api_name": name, "arguments": left},
                          {"api_name": name, "arguments": right}, [api])

    def test_numeric_representation_normalized_but_values_and_booleans_distinct(self):
        self.assertTrue(self.agrees("Weather", {"lat": 74.006}, {"lat": "74.0060"}, {"lat": "float"}))
        self.assertTrue(self.agrees("Lookup", {"count": 3}, {"count": 3.0}, {"count": "int"}))
        self.assertFalse(self.agrees("Weather", {"lat": 74.006}, {"lat": -74.006}, {"lat": "float"}))
        self.assertFalse(self.agrees("Lookup", {"count": 1}, {"count": True}, {"count": "int"}))
        self.assertFalse(self.agrees("Lookup", {"count": 1.2}, {"count": "1.2"}, {"count": "int"}))

    def test_identifiers_and_passwords_are_not_numeric_fields(self):
        for key in ("id", "password"):
            self.assertFalse(self.agrees("Lookup", {key: "00123"}, {key: "123"}, {key: "str"}))
            self.assertFalse(self.agrees("Lookup", {key: "123"}, {key: 123}, {key: "str"}))

    def test_time_and_supported_address_punctuation_normalization(self):
        self.assertTrue(self.agrees("EmailReminder", {"time": "2021-01-01 10:00"},
                                   {"time": "2021-01-01 10:00:00"}, {"time": "str"}))
        self.assertFalse(self.agrees("EmailReminder", {"time": "2021-01-01 10:00"},
                                    {"time": "2021-01-01 11:00"}, {"time": "str"}))
        self.assertTrue(self.agrees("PersonalInfoUpdate", {"address": "123 Main St."},
                                   {"address": "123 Main St"}, {"address": "str"}))
        self.assertFalse(self.agrees("PersonalInfoUpdate", {"address": "123 Main St"},
                                    {"address": "124 Main St"}, {"address": "str"}))

    def test_calculator_semantics_are_bounded_arithmetic_not_string_identity(self):
        self.assertTrue(self.agrees("Calculator", {"formula": "10+20+30"},
                                   {"formula": "(30 + 10) + 20"}, {"formula": "str"}))
        self.assertFalse(self.agrees("Calculator", {"formula": "10+20+30"},
                                    {"formula": "10+20"}, {"formula": "str"}))
        self.assertFalse(self.agrees("Calculator", {"formula": "60"},
                                    {"formula": "__import__('os').getcwd()"}, {"formula": "str"}))

    def test_email_wording_excluded_but_structured_identity_is_required(self):
        original = {"recipient": "tom@example.com", "time": "2021-01-01 10:00:00",
                    "location": "Room 1", "content": "Meeting at Room 1 at ten."}
        types = {key: "str" for key in original}
        reworded = dict(original, content="Reminder: your meeting is scheduled.")
        self.assertTrue(self.agrees("EmailReminder", original, reworded, types))
        self.assertFalse(self.agrees("EmailReminder", original, dict(reworded, content=""), types))
        for key, value in (("recipient", "mary@example.com"), ("time", "2021-01-01 11:00:00"),
                           ("location", "Room 2")):
            self.assertFalse(self.agrees("EmailReminder", original, dict(reworded, **{key: value}), types))

    def test_other_api_missing_action_or_changed_fields_do_not_agree(self):
        action = {"api_name": "Lookup", "arguments": {"id": "001"}}
        self.assertFalse(equivalent(action, None, [document()]))
        self.assertFalse(equivalent(action, dict(action, api_name="AnotherAPI"), [document()]))
        self.assertFalse(equivalent(action, dict(action, arguments={}), [document()]))
        self.assertFalse(equivalent(action, dict(action, arguments={"id": "001", "extra": 1}), [document()]))

    def test_email_exclusion_is_explicit_in_frozen_protocol(self):
        task, trajectory = fixtures()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            data = base / "dataset"
            (data / "test-data").mkdir(parents=True)
            (data / "test-data/level-3.json").write_text(json.dumps([task]))
            source = base / "trajectories.jsonl"
            source.write_text(json.dumps(trajectory) + "\n")
            _, protocol = prepare(base / "out", 3, source=source, data_dir=data)
        self.assertIn("EmailReminder.content ignored", protocol["consistency_definition"])
        self.assertIn("structured fields retained", protocol["consistency_definition"])
        self.assertFalse(protocol["labels_used_for_requests"])
        self.assertEqual(protocol["new_tool_executions"], 0)


class StrictSelfReportParsingTests(unittest.TestCase):
    def test_only_standalone_probability_is_accepted(self):
        for text, expected in (("0", 0.0), ("1", 1.0), (" 0.82\n", .82), (".75", .75),
                               ("```json\n0.9\n```", .9)):
            self.assertEqual(parse_self(text), expected, text)
        for text in ("confidence is 0.9", "I checked 3 fields; 0.9", "90%", "90", "1.01", "-0.1",
                     "NaN", "Infinity", '{"confidence":0.8}', "0.8 0.2", "", "0.8/1.0"):
            self.assertIsNone(parse_self(text), text)


if __name__ == "__main__":
    unittest.main()
