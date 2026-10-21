import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from calibrate_consistency import (assign_folds, budget_selection, fit_sigmoid,
                                   oof_calibrate, predict_json, score_metrics)


def fixtures():
    rows = []
    for task in range(15):
        for step, y, score in [(1, 0, 0.), (2, 1, 1.), (3, 1, 2/3)]:
            rows.append({"task_id": f"task-{task}", "step_id": step, "y": y, "consistency": score})
    return rows


class ConsistencyCalibrationTests(unittest.TestCase):
    def test_folds_keep_whole_tasks_and_oof_prior_uses_only_training_labels(self):
        rows = fixtures()
        records, folds, _, _ = oof_calibrate(rows)
        self.assertEqual(len(records), len(rows))
        by_task = {}
        for record in records:
            by_task.setdefault(record["task_id"], set()).add(record["fold_id"])
        self.assertTrue(all(len(fold_ids) == 1 for fold_ids in by_task.values()))
        self.assertEqual(len(folds), 5)
        for fold in folds:
            self.assertFalse(set(fold["train_tasks"]) & set(fold["valid_tasks"]))
            expected_prior = fold["train_correct"]/fold["train_n"]
            for record in records:
                if record["fold_id"] == fold["fold_id"]:
                    self.assertEqual(record["p_correct_prior"], expected_prior)
                    self.assertAlmostEqual(record["p_correct_oof"], predict_json(fold["fit"], record["raw_consistency"]))

    def test_json_mapping_is_monotone_and_missing_is_not_a_zero_score(self):
        model = fit_sigmoid([0., 0., 1., 1.], [0, 0, 1, 1])
        loaded = json.loads(json.dumps(model))
        self.assertGreater(loaded["coefficient"], 0.)
        probabilities = [predict_json(loaded, value) for value in [0., 1/3, 2/3, 1.]]
        self.assertEqual(probabilities, sorted(probabilities))
        self.assertIsNone(predict_json(loaded, None))
        self.assertGreater(predict_json(loaded, 0.), 0.)
        for invalid in [True, float("nan"), -.1, 1.1]:
            with self.assertRaises(ValueError):
                predict_json(loaded, invalid)
        with self.assertRaises(ValueError):
            predict_json({**loaded, "coefficient": -1}, .5)

    def test_reversed_training_signal_falls_back_to_prior_without_flipping(self):
        model = fit_sigmoid([0., 0., 1., 1.], [1, 1, 0, 0])
        self.assertLess(model["unconstrained_coefficient"], 0.)
        self.assertEqual(model["fit_status"], "intercept_only_nonpositive_slope")
        self.assertEqual(model["coefficient"], 0.)
        self.assertEqual(predict_json(model, 0.), .5)
        self.assertEqual(predict_json(model, 1.), .5)

    def test_budget_has_fixed_integer_size_and_reproducible_tie_selection(self):
        scores = [.5]*198
        selected = budget_selection(scores)
        self.assertEqual(int(selected.sum()), 39)
        np.testing.assert_array_equal(selected, budget_selection(scores))
        self.assertFalse(np.array_equal(selected, budget_selection(scores, seed=123)))

    def test_reliability_and_nll_handle_valid_boundary_probabilities(self):
        stat = score_metrics([0., 1.], [0, 1])
        self.assertEqual(stat["AUROC"], 1.)
        self.assertEqual(stat["Brier"], 0.)
        self.assertEqual(stat["ECE"], 0.)
        self.assertTrue(np.isfinite(stat["NLL"]))
        self.assertEqual(sum(b["n"] for b in stat["bins"]), 2)

    def test_fixed_real_development_folds_include_both_classes(self):
        labels = [json.loads(s) for s in (ROOT/"outputs/legacy_binary_v2/calls_binary.jsonl").read_text().splitlines()]
        rows = [{"task_id": r["task_id"], "y": r["revision"]["y"]} for r in labels]
        folds = assign_folds(rows)
        self.assertEqual(len(folds), 5)
        self.assertEqual(sum(len(valid) for _, _, valid in folds), 198)


if __name__ == "__main__":
    unittest.main()
