import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from analyze_verification import paired_delta_ci, summarize  # noqa: E402


class VerificationAnalysisTests(unittest.TestCase):
    def test_grounded_labels_take_priority_over_lucky_replay_match(self):
        traj = {
            "task_id": "t1", "seed": 0, "final_success": False,
            "replay_success": True, "total_verification_calls": 1,
            "steps": [{
                "action_type": "tool_call", "verified": True,
                "candidate_matched_gold": True,
                "candidate_grounded_correct": False,
                "executed_grounded_correct": False,
                "signal_model_calls": 0, "verifier_model_calls": 1,
            }],
        }
        out = summarize([traj])
        self.assertEqual(out["trigger_label_basis"], "grounded")
        self.assertEqual(out["trigger_precision"], 1.0)
        self.assertEqual(out["replay_trigger_precision"], 0.0)
        self.assertEqual(out["replay_success_rate"], 1.0)

    def test_summary_counts_fix_and_harm(self):
        traj = {
            "task_id": "t1", "seed": 0, "final_success": True,
            "clean_success": False, "recovered_success": True,
            "total_tokens": 100, "total_model_calls": 3,
            "total_verification_calls": 1, "total_tool_calls": 2,
            "steps": [
                {"action_type": "tool_call", "verified": True,
                 "candidate_matched_gold": False, "correction_outcome": "error_to_correct"},
                {"action_type": "tool_call", "verified": True,
                 "candidate_matched_gold": True, "correction_outcome": "correct_to_error"},
                {"action_type": "answer", "verified": False},
            ],
        }
        out = summarize([traj])
        self.assertEqual(out["n_steps"], 2)
        self.assertEqual(out["error_to_correct"], 1)
        self.assertEqual(out["correct_to_error"], 1)
        self.assertEqual(out["trigger_precision"], 0.5)
        self.assertEqual(out["correction_precision"], 0.5)

    def test_paired_comparison_uses_task_and_seed(self):
        base = [
            {"task_id": "a", "seed": 0, "final_success": False},
            {"task_id": "b", "seed": 0, "final_success": True},
        ]
        method = [
            {"task_id": "a", "seed": 0, "final_success": True},
            {"task_id": "b", "seed": 0, "final_success": True},
        ]
        out = paired_delta_ci(base, method, n_boot=100, seed=0)
        self.assertEqual(out["n_paired"], 2)
        self.assertEqual(out["wins"], 1)
        self.assertEqual(out["losses"], 0)
        self.assertEqual(out["success_delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
