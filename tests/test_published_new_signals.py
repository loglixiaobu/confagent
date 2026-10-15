"""Check the frozen public scores against compact samples, contexts and labels."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from collect_new_signals import digest, equivalent, parse_self

OUT = ROOT / "outputs/new_signals_fast_v1"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row):
    return row["task_id"], row["step_id"]


class PublishedNewSignalsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls = rows(ROOT / "outputs/legacy_binary_v2/calls_binary.jsonl")
        cls.states = rows(OUT / "states.jsonl")
        cls.scores = rows(OUT / "scores.jsonl")
        cls.samples = rows(OUT / "sample_actions.jsonl")
        cls.by_call = {key(r): r for r in cls.calls}
        cls.by_state = {key(r): r for r in cls.states}
        cls.by_sample = {key(r): r for r in cls.samples}

    def test_all_frozen_calls_have_one_self_report_and_three_independent_samples(self):
        wanted = set(self.by_call)
        for records in (self.calls, self.states, self.scores, self.samples):
            self.assertEqual(len(records), 198)
            self.assertEqual({key(r) for r in records}, wanted)
        self.assertEqual(sum(r["revision"]["y"] for r in self.calls), 158)
        seeds = []
        for row in self.samples:
            self.assertEqual([s["sample_index"] for s in row["samples"]], [0, 1, 2])
            seeds.extend([row["self_report_seed"]] + [s["seed"] for s in row["samples"]])
        self.assertEqual(len(seeds), 792)
        self.assertEqual(len(set(seeds)), 792)

    def test_compact_responses_reproduce_every_published_score(self):
        no_tool = multiple = truncated = 0
        for score in self.scores:
            state, responses = self.by_state[key(score)], self.by_sample[key(score)]
            candidate = state["candidate"]
            self.assertEqual(candidate, self.by_call[key(score)]["action"])
            self.assertEqual(score["state_sha256"], digest(state))
            self.assertEqual(score["self_report"], parse_self(responses["self_report_text"]))
            sampled = responses["samples"]
            agree = sum(equivalent(candidate, s["action"], state["api_documents"]) for s in sampled)
            strict = sum(s["action"] == candidate for s in sampled)
            self.assertEqual(score["n_samples"], 3)
            self.assertEqual(score["n_agree"], agree)
            self.assertEqual(score["n_strict_agree"], strict)
            self.assertEqual(score["consistency"], agree / 3)
            self.assertEqual(score["strict_consistency"], strict / 3)
            no_tool += sum(s["action"] is None for s in sampled)
            multiple += sum(s["tool_call_count"] > 1 for s in sampled)
            truncated += sum(s["finish_reason"] == "length" for s in sampled)
        self.assertEqual((no_tool, multiple, truncated), (29, 8, 0))
        summary = json.loads((OUT / "collection_summary.json").read_text())
        self.assertEqual(summary["no_tool_samples"], no_tool)
        self.assertEqual(summary["multiple_tool_samples"], multiple)
        self.assertEqual(summary["samples"], 594)
        self.assertEqual(summary["successful_requests"], 792)

    def test_model_contexts_only_contain_results_from_prior_steps(self):
        trajectories = {r["task_id"]: r for r in rows(ROOT / "results/traj_main_clean.jsonl")}
        fields = {"task_id", "step_id", "candidate", "messages", "tools", "api_documents"}
        for state in self.states:
            self.assertEqual(set(state), fields)
            prior = trajectories[state["task_id"]]["steps"][:state["step_id"] - 1]
            observed = [m["content"] for m in state["messages"] if m["role"] == "tool"]
            expected = [json.dumps(s["observation"], ensure_ascii=False, default=str)[:4000] for s in prior]
            self.assertEqual(observed, expected)
            self.assertEqual(len(observed), state["step_id"] - 1)

    def test_auc_and_provenance_match_frozen_public_artifacts(self):
        metrics = json.loads((OUT / "new_signal_metrics.json").read_text())
        protocol = json.loads((OUT / "protocol.json").read_text())
        scored = {key(r): r for r in self.scores}
        y = [r["revision"]["y"] for r in self.calls]
        expected = {"logprob": 0.635126582278481, "self_report": 0.7171677215189873,
                    "consistency": 0.8811708860759493}
        for name, value in expected.items():
            p = [(r["original_step"]["raw_logprobs"]["geo_mean_prob"] if name == "logprob"
                  else scored[key(r)][name]) for r in self.calls]
            self.assertAlmostEqual(roc_auc_score(y, p), value, places=12)
            self.assertEqual(metrics["cohorts"]["all_calls"]["common_all"]["signals"][name]["AUROC"], value)
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(protocol["states_sha256"], digest(self.states))
        self.assertEqual(protocol["source_sha256"], sha(ROOT / "results/traj_main_clean.jsonl"))
        self.assertEqual(protocol["public_collector_sha256"], sha(ROOT / "code/collect_new_signals.py"))
        self.assertEqual(metrics["evaluation_code_sha256"], sha(ROOT / "code/evaluate_new_signals.py"))
        self.assertEqual(metrics["scores_sha256"], sha(OUT / "scores.jsonl"))


if __name__ == "__main__":
    unittest.main()
