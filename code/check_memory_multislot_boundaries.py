#!/usr/bin/env python
"""Validate the V9 boundary development/test cohort and print a summary."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from memory_multislot_boundary_data import (
    DEFAULT_MULTISLOT_BOUNDARY_CASES,
    load_boundary_candidate_replay,
    load_boundary_model_inputs,
    load_multislot_boundary_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_MULTISLOT_BOUNDARY_CASES)
    args = parser.parse_args()

    model_inputs = load_boundary_model_inputs(args.cases)
    evaluator_cases = load_multislot_boundary_cases(args.cases)
    candidates = load_boundary_candidate_replay(args.cases)
    split_counts = Counter(row["dataset_split"] for row in evaluator_cases)
    categories = {
        split: Counter(row["boundary_category"] for row in evaluator_cases
                       if row["dataset_split"] == split)
        for split in split_counts
    }
    labels = Counter(row["expected_label"] for row in candidates)
    print("V9 boundary extension: PASS")
    print("cases:", len(evaluator_cases), dict(sorted(split_counts.items())))
    print("gold-free model inputs:", len(model_inputs))
    print("categories:", {split: dict(sorted(counts.items()))
                           for split, counts in sorted(categories.items())})
    print("development offline candidates:", len(candidates), dict(sorted(labels.items())))
    print("test pre-generated candidates: 0")
    print("tools:", ", ".join(case["target_tool"] for case in evaluator_cases))


if __name__ == "__main__":
    main()
