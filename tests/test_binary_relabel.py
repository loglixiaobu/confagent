"""Offline regression tests using compact benchmark fixtures and frozen calls."""
import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from apibank_env import APIBankMockEnv, api_schema_to_tool, arithmetic_value
from offline_confidence_utils import capture, schema_valid, sha, summarize_cohort, tied_risk
from relabel_binary_steps import final_body, label_call, task_outcome
from evaluate_binary_logprob import checked_inputs


class BinaryRelabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls, cls.ends, cls.tasks, cls.summary = checked_inputs(ROOT/'outputs/legacy_binary_v2')
        cls.tasks = {r['task_id']: r for r in cls.tasks}
        cls.by_key = {(r['task_id'], r['step_id']): r for r in cls.calls}
        cls.source = ROOT/'results/traj_main_clean.jsonl'
        cls.original = {r['task_id']: r for r in map(json.loads, cls.source.read_text().splitlines())}
        cls.fixtures = json.loads((ROOT/'tests/fixtures/label_cleaning_cases.json').read_text())['tasks']

    def decide(self, ti, sid, step_override=None):
        tid = f'l3-{ti:03d}'
        recorded = self.original[tid]['steps']
        step = step_override or next(s for s in recorded if s['step_id'] == sid)
        prior = [{'step_id': s['step_id'], 'action': s['action_input'], 'observation': s['observation']}
                 for s in recorded if s['step_id'] < sid]
        return label_call(ti, self.fixtures[str(ti)], step, prior, tid)

    def test_all_calls_are_binary_and_original_fields_unchanged(self):
        self.assertEqual(len(self.calls), 198)
        self.assertEqual(sum(r['revision']['y'] for r in self.calls), 158)
        for row in self.calls:
            self.assertIn(row['revision']['y'], (0, 1))
            old = next(s for s in self.original[row['task_id']]['steps'] if s['step_id'] == row['step_id'])
            self.assertEqual(row['original_step'], old)

    def test_fabricated_inputs_fail_but_relevant_search_passes(self):
        self.assertEqual(self.decide(35, 1)['y'], 0)
        self.assertEqual(self.decide(35, 3)['y'], 1)
        self.assertEqual(self.decide(37, 2)['y'], 0)
        self.assertEqual(self.decide(10, 6)['y'], 0)

    def test_numeric_artifacts_are_repaired_without_accepting_wrong_units(self):
        self.assertEqual(self.decide(7, 1)['y'], 1)
        self.assertEqual(self.decide(16, 2)['y'], 1)
        self.assertEqual(self.decide(16, 5)['y'], 0)

    def test_salary_conversion_needs_evidence(self):
        self.assertEqual(self.decide(41, 2)['y'], 0)
        self.assertEqual(self.decide(41, 3)['y'], 1)

    def test_missing_counts_do_not_get_filled_with_future_returns(self):
        self.assertEqual(self.decide(26, 4)['y'], 0)
        self.assertEqual(self.decide(26, 7)['y'], 1)
        self.assertEqual(self.decide(27, 5)['y'], 1)
        self.assertEqual(arithmetic_value('70+80+90'), 240)

    def test_labels_do_not_depend_on_confidence(self):
        old = self.by_key[('l3-035', 1)]['original_step']
        changed = copy.deepcopy(old)
        changed['confidence'] = .001
        changed['raw_logprobs'] = {'available': False, 'geo_mean_prob': .999}
        self.assertEqual(self.decide(35, 1, old), self.decide(35, 1, changed))

    def test_missing_work_is_attached_to_end_not_correct_prior_calls(self):
        self.assertTrue(all(r['revision']['y'] == 1 for r in self.calls if r['task_id'] == 'l3-000'))
        self.assertEqual(self.tasks['l3-000']['y'], 0)
        self.assertIn('Room 2', ' '.join(self.tasks['l3-000']['missing_goals']))

    def test_actual_completion_allows_recovery_and_direct_updates(self):
        self.assertEqual(self.tasks['l3-026']['y'], 1)
        self.assertEqual(self.tasks['l3-045']['y'], 1)
        self.assertEqual(self.tasks['l3-007']['y'], 0)
        self.assertEqual(sum(r['y'] for r in self.tasks.values()), 14)

    def test_task_checks_reproduce_from_fixtures(self):
        for ti in (0, 8, 10, 26, 35, 37, 41, 45):
            tid = f'l3-{ti:03d}'
            calls = [r for r in self.calls if r['task_id'] == tid]
            outcome = task_outcome(ti, self.fixtures[str(ti)], self.original[tid], calls)
            self.assertEqual(outcome, self.tasks[tid])

    def test_no_fake_stop_actions_or_scores(self):
        self.assertEqual(len(self.ends), 18)
        self.assertEqual(sum(r['y'] for r in self.ends), 9)
        self.assertTrue(all(not r['score_available'] for r in self.ends))
        self.assertTrue(all('confidence' not in r and 'raw_logprobs' not in r for r in self.ends))
        self.assertFalse(any(r['task_id'] == 'l3-020' for r in self.ends))
        for tid in ('l3-011', 'l3-012', 'l3-024'):
            self.assertEqual(self.tasks[tid]['y'], 0)

    def test_original_logprob_metrics_are_reproducible(self):
        stat = summarize_cohort([(r, r['revision']['y']) for r in self.calls])
        self.assertAlmostEqual(stat['signals']['geo_mean_prob']['AUROC'], .635126582278481)
        dedup = summarize_cohort([(r, r['revision']['y']) for r in self.calls if r['first_semantic_attempt']])
        self.assertAlmostEqual(dedup['signals']['geo_mean_prob']['AUROC'], .698)

    def test_schema_and_score_ties(self):
        doc = APIBankMockEnv(self.fixtures['10']).tool_registry['AddMeeting']
        schema = api_schema_to_tool(doc)['function']['parameters']['properties']['attendees']
        self.assertTrue(schema_valid(['Mary', 'Peter'], schema))
        self.assertFalse(schema_valid('Mary, Peter', schema))
        p, y = np.full(5, .8), np.array([1, 1, 0, 0, 0])
        self.assertAlmostEqual(capture(p, y, .2), .2)
        np.testing.assert_allclose(tied_risk(p, y)[1], .6)

    def test_unclosed_thinking_is_not_a_final_answer(self):
        self.assertEqual(final_body('<think>still planning'), ('', True))
        self.assertEqual(final_body('<think>reason</think>Done'), ('Done', False))


if __name__ == '__main__': unittest.main()
