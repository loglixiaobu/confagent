"""Critical Q3 execution contracts; no experiment snapshots or model calls."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from q3_three_layer.run import Experiment, main
from q3_three_layer.backend import checkpoint_hashes
from q3_three_layer.tasks import TaskWorld, build_suite, computational_verify

class FakeActor:
    def __init__(self, prediction):
        self.prediction = prediction
        self.calls = []

    def generate(self, messages, adapter=False):
        self.calls.append({"messages": messages, "adapter": adapter})
        return {"prediction": self.prediction, "key": "fake-test-only", "raw": "fixture",
                "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1,
                "latency_s": 0, "cache_hit": False}

class RecordingPolicy:
    def __init__(self, choice):
        self.choice = choice
        self.seen = []

    def choose(self, vector, available):
        self.seen.append((vector, available))
        return self.choice

class ThreeLayerFlowTests(unittest.TestCase):
    def test_entry_uses_standard_checkpoint_and_no_private_files(self):
        # Run from an empty directory: the protocol must live in the package.
        suite = build_suite()
        predictions = {str(TaskWorld(task).prepare()["messages"]):
                       computational_verify(TaskWorld(task).prepare()["contract"])["prediction"]
                       for task in suite["construction"]}

        class ConstructionActor(FakeActor):
            def __init__(self, *args, **kwargs):
                super().__init__(None)

            def generate(self, messages, adapter=False):
                self.prediction = predictions[str(messages)]
                return super().generate(messages, adapter)

        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                checkpoint = Path(directory) / "model"
                checkpoint.mkdir()
                (checkpoint / "config.json").write_text("{}", encoding="utf-8")
                weight = checkpoint / "model.safetensors"
                weight.write_bytes(b"test-fixture-not-a-real-model")
                before = checkpoint_hashes(checkpoint)
                weight.write_bytes(b"changed-test-fixture")
                self.assertNotEqual(before, checkpoint_hashes(checkpoint))
                with patch("q3_three_layer.run.HFActor", ConstructionActor), patch.object(
                        sys, "argv", ["run", "--model", "unused-checkpoint", "--out", "result", "--construction-only"]):
                    main()
                import json
                output = Path(directory) / "result"
                protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
                self.assertEqual(protocol["protocol_version"], 2)
                self.assertEqual(protocol["memory"]["admission_unit"], "conversion_identity_bundle")
                self.assertEqual(len(json.loads((output / "source_experiences.json").read_text(encoding="utf-8"))), 18)
            finally:
                os.chdir(original)

    def test_source_recovery_is_actual_overwrite_and_honestly_labeled(self):
        task = build_suite()["construction"][0]
        actor = FakeActor({"value": "-999", "unit": "m"})
        with tempfile.TemporaryDirectory() as directory:
            exp = Experiment(actor, directory)
            row = exp.collect([task])[0]
            self.assertFalse(row["first"]["success"])
            self.assertTrue(row["final"]["success"])
            self.assertEqual(row["origin"], "tool_assisted_repair")
            self.assertEqual(row["final"]["wrong_writes"], 1)
            self.assertEqual(row["final"]["overwrites"], 1)
            self.assertTrue(row["final"]["recovered_success"])
            self.assertEqual(exp.computational_calls, 1)
            self.assertTrue(row["certified"])

    def test_chain_initial_draft_is_readonly_and_only_one_read_and_write_occur(self):
        task = build_suite()["chain_test"][0]
        actor = FakeActor({"value": "-999", "unit": "m"})
        policy = RecordingPolicy("verify")
        with tempfile.TemporaryDirectory() as directory:
            row = Experiment(actor, directory).evaluate([task], [], policy, p=True)[0]
            self.assertTrue(row["success"])
            self.assertEqual([e["kind"] for e in row["history"]], ["read", "write"])
            self.assertEqual(row["assessment"]["environment_calls"], 2)
            self.assertEqual(row["assessment"]["wrong_writes"], 0)
            self.assertEqual(row["verifier_calls"], 1)
            self.assertEqual(len(actor.calls), 1)
            self.assertNotIn("memory", policy.seen[0][1])

    def test_bundle_requires_actual_memory_hits_for_conversion_and_identity(self):
        from q3_three_layer.run import admit_memory

        tasks = build_suite()["development"]

        class GuidedCorrectActor(FakeActor):
            def generate(self, messages, adapter=False):
                task = next(t for t in tasks if t["requirement"] in messages[1]["content"])
                contract = TaskWorld(task).prepare()["contract"]
                self.prediction = (computational_verify(contract)["prediction"] if len(messages) > 2
                                   else {"value": contract["source_value"],
                                         "unit": contract["target_unit"]})
                return super().generate(messages, adapter)

        conversion = {"source_unit": "km", "target_unit": "m", "multiplier": 1000,
                      "instruction": "Convert the CURRENT km quantity to m by multiplying by 1000."}
        identity = {"source_unit": "m", "target_unit": "m", "multiplier": 1,
                    "instruction": "Keep the CURRENT m quantity unchanged in m."}
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment(GuidedCorrectActor(None), directory)
            baseline = experiment.evaluate(tasks, [])
            partial = admit_memory(experiment, tasks, [conversion], False, baseline)
            self.assertEqual(partial["accepted"], [])
            self.assertFalse(partial["bundles"]["m"]["pass"])
            self.assertEqual(partial["bundles"]["m"]["reason"],
                             "missing_conversion_identity_coverage")
            complete = admit_memory(experiment, tasks, [conversion, identity], False, baseline)
            self.assertEqual(len(complete["accepted"]), 2)
            self.assertTrue(complete["bundles"]["m"]["pass"])
            self.assertEqual((complete["bundles"]["m"]["F"],
                              complete["bundles"]["m"]["H"]), (2, 0))
