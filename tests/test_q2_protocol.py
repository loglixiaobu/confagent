import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from apibank_env import load_level3  # noqa: E402
from react_agent import AgentConfig  # noqa: E402
from run_batch import (create_task_manifest, load_task_manifest,  # noqa: E402
                       task_fingerprint)


def tasks_fixture():
    return [
        {"requirement": "A", "apis": [{"api_name": "A", "input": {"x": 1}}]},
        {"requirement": "B", "apis": [{"api_name": "B", "input": {"x": 2}}]},
        {"requirement": "C", "apis": [{"api_name": "C", "input": {"x": 3}}]},
    ]


class Q2ProtocolTests(unittest.TestCase):
    def test_manifest_round_trip_and_tamper_detection(self):
        tasks = tasks_fixture()
        manifest = create_task_manifest(tasks, [0, 2], "test", "development")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "manifest.json")
            with open(path, "w", encoding="utf-8") as output:
                json.dump(manifest, output)
            loaded, indices = load_task_manifest(path, tasks, "test")
            self.assertEqual(indices, [0, 2])
            self.assertEqual(loaded["manifest_id"], manifest["manifest_id"])

            changed = tasks_fixture()
            changed[2]["requirement"] = "changed"
            with self.assertRaisesRegex(ValueError, "数据集指纹"):
                load_task_manifest(path, changed, "test")

    def test_private_metadata_does_not_change_task_fingerprint(self):
        task = tasks_fixture()[0]
        tagged = {**task, "_dataset_index": 10, "_task_id": "test-00010"}
        self.assertEqual(task_fingerprint(task), task_fingerprint(tagged))

    def test_unknown_dataset_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "train 或 test"):
            load_level3("unused", "validation")

    def test_core_policies_enforce_shared_verifier_and_minimum_budget(self):
        with self.assertRaisesRegex(ValueError, "至少需要"):
            AgentConfig(verify_policy="confidence_check", verifier="critic",
                        verify_budget=3)
        with self.assertRaisesRegex(ValueError, "固定使用 critic"):
            AgentConfig(verify_policy="random_check", verifier="consistency",
                        verify_budget=4)
        config = AgentConfig(verify_policy="confidence_check", verifier="critic",
                             verify_budget=4)
        self.assertEqual(config.max_checks_per_task, 1)
        with self.assertRaisesRegex(ValueError, "禁止额外"):
            AgentConfig(verify_policy="direct", verify_budget=0, self_report=True)


if __name__ == "__main__":
    unittest.main()
