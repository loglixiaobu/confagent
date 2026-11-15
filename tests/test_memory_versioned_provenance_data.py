"""Check linked authorization evidence and independence of the V8 cases."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from memory_versioned_provenance_data import DEFAULT_PROVENANCE_CASES, load_provenance_cases


class ProvenanceMemoryDataTest(unittest.TestCase):
    def load_mutated(self, change):
        data = json.loads(DEFAULT_PROVENANCE_CASES.read_text(encoding="utf-8"))
        change(data["replication"][0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return load_provenance_cases(path)

    def test_earlier_source_names_the_requested_natural_detail(self):
        cases = load_provenance_cases()
        self.assertEqual({split: len(rows) for split, rows in cases.items()},
                         {"validation": 8, "replication": 16, "updated": 20})
        for split in ("replication", "updated"):
            for row in cases[split]:
                with self.subTest(case=row["task_id"]):
                    saved, = [event for event in row["memory_events"]
                              if event["record_id"] in row["permitted_memory_ids"]]
                    detail = row["provenance"]["natural_detail"]
                    self.assertIn(detail.casefold(), row["requirement"].casefold())
                    self.assertIn(detail.casefold(), saved["observation"][0]["content"].casefold())
                    self.assertEqual(row["history"], [])

    def test_bare_value_without_named_detail_is_rejected(self):
        def change(row):
            record = row["memory_events"][0]
            record["observation"][0]["content"] = "Please save this value: " + record["value"] + "."
        with self.assertRaisesRegex(ValueError, "does not identify its natural slot"):
            self.load_mutated(change)

    def test_memory_value_leak_into_current_prompt_is_rejected(self):
        def change(row):
            row["requirement"] += " The value is " + row["memory_events"][0]["value"] + "."
        with self.assertRaisesRegex(ValueError, "must occur only in prior evidence"):
            self.load_mutated(change)
