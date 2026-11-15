"""Small handoff API: pre-call confidence, execution feedback, review budget.

No automatic repair and no gold/reference labels are read by this module.
Create one ReviewBudget per task. Execution flags do not assert model errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from confidence_runtime import estimate_step_confidence, load_calibrator


class StepConfidenceMonitor:
    def __init__(self, calibrator, *, client=None, request_fn=None, before_request=None):
        self.calibrator = load_calibrator(calibrator)
        self.client, self.request_fn, self.before_request = client, request_fn, before_request

    def before_call(self, *, messages, tools, candidate, api_documents,
                    task_id: str, step_id: int) -> dict[str, Any]:
        """Use only history available BEFORE this candidate is executed."""
        if not isinstance(task_id, str) or not task_id or type(step_id) is not int or step_id < 1:
            raise ValueError('A nonempty task_id and positive integer step_id are required')
        confidence = estimate_step_confidence(
            messages=messages, tools=tools, candidate=candidate, api_documents=api_documents,
            calibrator=self.calibrator, client=self.client, request_fn=self.request_fn,
            before_request=self.before_request, request_namespace=f'{task_id}:{step_id}')
        return self.summarize(confidence, task_id=task_id, step_id=step_id)

    def summarize(self, confidence, *, task_id, step_id):
        """Compact a live or archived runtime output without changing its score."""
        return {'schema_version': 'step-monitor-v1', 'stage': 'pre_call',
                'task_id': task_id, 'step_id': step_id,
                'raw_consistency': confidence['raw_consistency'],
                'p_correct': confidence['p_correct'],
                'needs_verification': confidence['needs_verification'],
                'reason': confidence['reason'],
                'score_status': 'missing' if confidence['p_correct'] is None else 'available',
                'suggested_checks': ['parameter_source', 'units', 'preconditions']
                    if confidence['needs_verification'] else [],
                'calibrator_id': self.calibrator.get('artifact_id', self.calibrator['version']),
                'cost': confidence.get('cost', {}), 'quality': confidence.get('quality', {})}


def execution_feedback(observation, *, exception=None):
    """Read the ACTUAL return; do not use it to rewrite pre-call probability."""
    flags = []
    if exception is not None:
        flags.append('execution_exception')
    if observation is None:
        flags.append('missing_return')
    elif observation == '' or observation == [] or observation == {}:
        flags.append('empty_return')
    if isinstance(observation, dict):
        # False/None/empty error fields do not indicate an error.
        if observation.get('error') or observation.get('exception'):
            flags.append('tool_reported_error')
        if observation.get('success') is False:
            flags.append('tool_reported_failure')
    return {'schema_version': 'step-monitor-v1', 'stage': 'post_call',
            'flags': flags, 'needs_verification': bool(flags),
            'reason': flags[0] if flags else 'no_structural_anomaly',
            'call_correctness': 'not_determined_by_execution_status'}


def ending_feedback(*, finish_reason=None, stop_reason=None, task_complete=None):
    """Completion may be supplied by a separate evaluator; never assume success."""
    if task_complete is not None and type(task_complete) is not bool:
        raise ValueError('task_complete must be True, False, or None')
    flags = []
    if finish_reason == 'length' or stop_reason == 'truncated_no_call':
        flags.append('generation_truncated')
    if stop_reason in ('max_steps', 'repeated_call'):
        flags.append(stop_reason)
    if task_complete is False:
        flags.append('task_incomplete')
    if task_complete is None:
        flags.append('completion_unassessed')
    return {'schema_version': 'step-monitor-v1', 'stage': 'ending',
            'flags': flags, 'needs_verification': bool(flags),
            'task_complete': task_complete,
            'reason': flags[0] if flags else 'completion_confirmed_by_caller'}


@dataclass
class ReviewBudget:
    """Reserve at most max_checks per task, shared by pre/post/ending checks."""
    max_checks: int = 1
    used: int = 0

    def __post_init__(self):
        if (type(self.max_checks) is not int or self.max_checks < 0
                or type(self.used) is not int or not 0 <= self.used <= self.max_checks):
            raise ValueError('Invalid per-task check budget')

    def claim(self, feedback):
        """Call immediately before the checker; a skipped request consumes nothing."""
        if feedback.get('needs_verification') is not True or self.used >= self.max_checks:
            return False
        self.used += 1
        return True
