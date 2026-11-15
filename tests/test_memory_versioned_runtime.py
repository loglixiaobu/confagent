"""Checks for revisions, revocation, and exact memory scope."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from run_memory_versioned import compile_facts, fact_rules, presented_case, used_stale_or_forbidden_value


def save(event_id, revision, value, profile="owner", slot="trip_city"):
    return {"record_id": event_id, "sequence": revision, "value": value,
            "profile_id": profile, "slot": slot, "kind": "save"}


class VersionedMemoryTests(unittest.TestCase):
    def test_revision_order_and_revocation(self):
        first = save("old", 1, "York")
        revoke = {"record_id": "clear", "sequence": 2, "profile_id": "owner",
                  "slot": "trip_city", "kind": "revoke"}
        other = save("another-profile", 4, "Bath", profile="other")
        current = save("current", 3, "Leeds")
        collect = lambda events: compile_facts({"test": [{"memory_events": events}]})
        self.assertEqual(collect([revoke, first, other])[("owner", "trip_city")], [])
        self.assertEqual(collect([current, other, first, revoke])[("owner", "trip_city")], [current])
        self.assertEqual(collect([current, first, other])[("owner", "trip_city")], [first, current])

    def test_replay_shares_latest_text_and_current_override_takes_priority(self):
        first = save("old", 1, "York")
        newest = save("new", 2, "Leeds")
        case = {
            "task_id": "test", "condition": "updated", "memory_field": "city",
            "target_tool": "Weather", "expected": {"kind": "call"},
            "parameter_sources": {"city": {"source": "history", "value": "Leeds"}},
            "api_documents": [{"name": "Weather", "input_parameters": {"city": {"type": "str"}}}],
            "memory_events": [first, newest],
        }
        rules = fact_rules(case, [newest])
        replay = fact_rules(case, [first, newest])
        self.assertEqual(rules, replay[-1:])
        self.assertNotIn("York", rules[0]["handling"])
        self.assertEqual(presented_case(case, [])["expected"], {"kind": "clarify", "fields": ["city"]})
        action = {"kind": "call", "api_name": "Weather", "arguments": {"city": "York"}}
        self.assertTrue(used_stale_or_forbidden_value(case, action))
        override = {**case, "parameter_sources": {"city": {"source": "requirement", "value": "York"}}}
        self.assertFalse(used_stale_or_forbidden_value(override, action))

    def test_source_linked_variant_keeps_the_same_value_and_uses_only_user_authorization(self):
        from run_memory_versioned_provenance import source_linked_rules

        event = {**save("confirmed", 2, "Leeds"), "observation": [
            {"role": "user", "content": "Please remember Leeds as the destination for my club trip."},
            {"role": "assistant", "content": "I recorded this."},
        ]}
        case = {"task_id": "test", "memory_field": "city",
                "api_documents": [{"name": "Weather", "input_parameters": {"city": {"type": "str"}}}]}
        plain, = fact_rules(case, [event])
        linked, = source_linked_rules(case, event)
        self.assertEqual(plain["rule_id"], linked["rule_id"])
        self.assertEqual(plain["applicability"], linked["applicability"])
        self.assertTrue(linked["handling"].endswith(plain["handling"]))
        self.assertIn(event["observation"][0]["content"], linked["handling"])
        self.assertNotIn(event["observation"][1]["content"], linked["handling"])
