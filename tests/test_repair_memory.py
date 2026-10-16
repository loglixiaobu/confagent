"""Critical repair evidence and harmful-memory rejection."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from repair_memory import RepairMemory, source_certificate, validate_transfer

def memory(name="one", status="proposed", api="Convert"):
    return RepairMemory(name, "Convert only when the source unit differs from the documented target unit.",
                        [api], [], ["construction-1"], 1,
                        {"eligible": True, "is_repair": True}, status=status)

def probes():
    return [{"task_id": "validation-" + str(i), "split": "validation",
             "variant": "conversion" if i < 2 else "already_target", "api_names": ["Convert"],
             "messages": [{"role": "user", "content": "Process the current visible value"}],
             "tools": [{"type": "function", "function": {"name": "Convert"}}]}
            for i in range(4)]

class RepairMemoryTests(unittest.TestCase):
    def test_source_certificate_does_not_trust_critic_or_replay_flags(self):
        certificate = source_certificate("construction-1", {"number": 1, "matched_gold": True},
                                         {"number": 2, "critic_valid": False},
                                         lambda action: action["number"] == 2)
        self.assertTrue(certificate["is_repair"])
        self.assertFalse(certificate["candidate_correct"])
        self.assertTrue(certificate["executed_correct"])

    def test_unknown_source_judgment_is_not_a_demonstrated_repair(self):
        result = source_certificate("c", "unknown", "correct", lambda a: None if a == "unknown" else True)
        self.assertIsNone(result["candidate_correct"])
        self.assertFalse(result["is_repair"])

    def test_transfer_gate_accepts_fix_but_rejects_harm_and_neutral_memory(self):
        judge = lambda probe, action: action == "correct"
        useful = lambda probe, selected: "correct" if selected or probe["variant"] == "already_target" else "wrong"
        item = memory()
        result = validate_transfer(item, probes(), useful, judge)
        self.assertEqual((result["fixes"], result["harms"], item.status), (2, 0, "accepted"))
        harmful = lambda probe, selected: ("wrong" if selected and probe["variant"] == "already_target"
                                           else "correct" if selected else "wrong" if probe["variant"] == "conversion" else "correct")
        item = memory()
        validate_transfer(item, probes(), harmful, judge)
        self.assertEqual(item.validation["harms"], 2)
        self.assertEqual(item.status, "rejected")
        item = memory()
        validate_transfer(item, probes(), lambda probe, selected: "correct", judge)
        self.assertEqual(item.status, "rejected")
