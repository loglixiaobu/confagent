"""Versioned experiment settings, independent of local research documents."""
from .backend import digest


def build_protocol(suite):
    return {
        "protocol_version": 2,
        "suite_sha256": digest(suite),
        "counts": {name: len(tasks) for name, tasks in suite.items()},
        "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "generation": {"max_new_tokens": 48, "do_sample": False, "enable_thinking": False},
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0, "steps": 12,
                 "learning_rate": 2e-4, "microbatch": 1, "max_sequence": 256,
                 "seed": 42, "target_modules": ["q_proj", "v_proj"]},
        "memory": {"source_entities_per_pair": 2,
                   "admission_unit": "conversion_identity_bundle",
                   "direct_pairs_per_member": 2, "pairs_per_bundle": 4,
                   "min_fixes": 1, "max_harms": 0, "require_actual_retrieval": True},
        "policy": {"ridge_lambda": 1.0, "verifier_slots": 1, "decisions": 1,
                   "verifier_cost": 0.10, "extra_model_cost": 0.02},
        "composition": {"development_pairs": 12, "max_harms": 0,
                        "require_positive_utility": True, "fallback": "base"},
        "target_access": "only after all selection; raw form prompts",
    }
