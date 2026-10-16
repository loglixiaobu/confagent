"""One identical HF backbone for base inference, real LoRA and cached inference.

Model libraries are imported lazily so the evidence and task tests run on CPU.
Raw requests remain local. All cache keys include adapter bytes, not just a flag.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import random
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def checkpoint_hashes(model_dir):
    """Identify standard HF files without requiring a machine-specific manifest."""
    root = Path(model_dir).resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError("HF checkpoint is missing config.json: " + str(root))
    weights = {p.name for pattern in ("*.safetensors", "pytorch_model*.bin") for p in root.glob(pattern)}
    if not weights:
        raise FileNotFoundError("HF checkpoint contains no model weights: " + str(root))
    indices = list(root.glob("*.index.json"))
    for index in indices:
        required = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
        if not required.issubset(weights):
            raise FileNotFoundError("HF checkpoint is missing indexed weight shards")
    names = weights | {p.name for p in indices}
    names |= {name for name in ("config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
        "vocab.json", "merges.txt") if (root / name).is_file()}
    return {name: file_digest(root / name) for name in sorted(names)}


class HFActor:
    def __init__(self, model_dir, output_dir, resume=False):
        import numpy as np
        import torch
        import transformers
        import peft
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.torch = torch
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output / "requests.jsonl"
        if self.log_path.exists() and not resume:
            raise FileExistsError("Use a fresh output directory or explicit --resume")
        self.cache = {}
        if resume and self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row
        assets = checkpoint_hashes(model_dir)
        self.identity = {"checkpoint_sha256": digest(assets), "assets": assets, "dtype": "bfloat16",
                         "torch": torch.__version__, "transformers": transformers.__version__,
                         "peft": peft.__version__, "max_new_tokens": 48, "do_sample": False,
                         "enable_thinking": False, "attention": "sdpa"}
        old = self.output / "actor_identity.json"
        if old.exists() and json.loads(old.read_text(encoding="utf-8")) != self.identity:
            raise ValueError("Actor identity changed during resume")
        write_json(old, self.identity)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa", local_files_only=True)
        self.model.eval()
        self.has_adapter = False
        self.adapter_hash = None
        self.actual_calls = 0
        self.cache_hits = 0
        print("HF actor loaded: " + json.dumps(self.identity), flush=True)

    def render(self, messages):
        from copy import deepcopy
        normalized = deepcopy(messages)
        for message in normalized:
            for call in message.get("tool_calls", []):
                arguments = call.get("function", {}).get("arguments")
                if isinstance(arguments, str):
                    # HF Qwen's template iterates argument items. The environment
                    # keeps the standard OpenAI JSON string in its raw history.
                    call["function"]["arguments"] = json.loads(arguments)
        return self.tokenizer.apply_chat_template(normalized, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    def adapter_context(self, active):
        if self.has_adapter and not active:
            return self.model.disable_adapter()
        return contextlib.nullcontext()

    def generate(self, messages, adapter=False, bypass_cache=False):
        from audited_client import wait_for_gpu
        from .tasks import parse_prediction

        if adapter and not self.has_adapter:
            raise ValueError("Requested untrained/unloaded adapter")
        identity = {**self.identity, "adapter": self.adapter_hash if adapter else None}
        key = digest({"identity": identity, "messages": messages})
        if not bypass_cache and key in self.cache:
            self.cache_hits += 1
            return {**self.cache[key], "cache_hit": True}
        thermal = wait_for_gpu()
        prompt = self.render(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        start = time.perf_counter()
        self.model.eval()
        with self.adapter_context(adapter), self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=48,
                use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
        self.torch.cuda.synchronize()
        ids = output[0, inputs.input_ids.shape[1]:].tolist()
        eos = self.tokenizer.eos_token_id
        finished = eos in ids or len(ids) < 48
        raw = self.tokenizer.decode(ids, skip_special_tokens=True)
        row = {"key": key, "identity": identity, "messages": messages, "raw": raw,
               "prediction": parse_prediction(raw) if finished else None,
               "finish_reason": "stop" if finished else "length",
               "prompt_tokens": int(inputs.input_ids.shape[1]),
               "completion_tokens": len(ids), "generated_ids": ids,
               "latency_s": time.perf_counter() - start, "thermal": thermal,
               "cache_hit": False}
        append_json(self.log_path, row)
        self.cache[key] = row
        self.actual_calls += 1
        print(f"generation {self.actual_calls} adapter={adapter} "
              f"tokens={len(ids)} output={raw[:160]!r}", flush=True)
        return row

    def parameter_hash(self, adapter_only=False):
        """Bounded host-memory checksum; avoids cloning a 1.7GB backbone."""
        h = hashlib.sha256()
        for name, parameter in self.model.named_parameters():
            is_adapter = "lora_" in name
            if is_adapter != adapter_only:
                continue
            # PEFT wraps keys. Compare before/after in the same wrapped model.
            h.update(name.encode())
            data = parameter.detach().contiguous().view(self.torch.uint8).flatten()
            for chunk in data.split(1024 * 1024):
                h.update(chunk.cpu().numpy().tobytes())
        return h.hexdigest()

    def train_adapter(self, examples, adapter_dir):
        import numpy as np
        import torch.nn.functional as F
        from peft import LoraConfig, get_peft_model
        from audited_client import wait_for_gpu

        torch = self.torch
        if self.has_adapter:
            raise ValueError("Training is one fixed candidate update, not repeated fitting")
        data = []
        rejected = []
        for item in examples:
            prompt_ids = self.tokenizer(self.render(item["messages"]),
                                         add_special_tokens=False)["input_ids"]
            target = json.dumps(item["prediction"], separators=(",", ":"))
            target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
            target_ids += [self.tokenizer.eos_token_id]
            if len(prompt_ids) + len(target_ids) > 256:
                rejected.append({"task_id": item["task_id"],
                                 "tokens": len(prompt_ids) + len(target_ids)})
                continue
            data.append({"task_id": item["task_id"], "prompt_ids": prompt_ids,
                         "target_ids": target_ids, "target": target})
        if not data:
            raise ValueError("No eligible training sequences fit the frozen token cap")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = get_peft_model(self.model, LoraConfig(r=4, lora_alpha=8,
            lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM", bias="none"))
        self.has_adapter = True
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        frozen_before = self.parameter_hash(False)
        adapter_before = self.parameter_hash(True)
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in n for n, _ in trainable):
            raise AssertionError("Only actual LoRA parameters may be trainable")
        initial = {n: p.detach().float().cpu().clone() for n, p in trainable}
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=2e-4)
        order = np.random.default_rng(42).permutation(len(data)).tolist()
        torch.cuda.reset_peak_memory_stats()
        steps = []
        start = time.perf_counter()
        self.model.train()
        for step in range(12):
            thermal = wait_for_gpu()
            item = data[order[step % len(data)]]
            ids = item["prompt_ids"] + item["target_ids"]
            tensor = torch.tensor([ids], dtype=torch.long, device="cuda")
            labels = torch.tensor([item["target_ids"]], dtype=torch.long, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter()
            # The retained logits begin one token before the assistant target.
            out = self.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                             use_cache=False, logits_to_keep=len(item["target_ids"]) + 1)
            logits = out.logits[:, :-1, :].float()
            if logits.shape[:2] != labels.shape:
                raise AssertionError("Assistant-only loss is misaligned")
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            grad_norm = float(torch.sqrt(sum((p.grad.detach().float().square().sum()
                for _, p in trainable if p.grad is not None), torch.zeros((), device="cuda"))).item())
            if not torch.isfinite(loss) or not np.isfinite(grad_norm) or grad_norm <= 0:
                raise AssertionError("Nonfinite loss or missing LoRA gradient")
            optimizer.step()
            torch.cuda.synchronize()
            row = {"step": step + 1, "task_id": item["task_id"],
                   "loss": float(loss.item()), "gradient_norm": grad_norm,
                   "target_tokens": len(item["target_ids"]),
                   "sequence_tokens": len(ids), "elapsed_s": time.perf_counter() - tick,
                   "thermal": thermal}
            steps.append(row)
            append_json(self.output / "training_steps.jsonl", row)
            print("LoRA step " + json.dumps(row), flush=True)
            del out, logits, loss, tensor, labels
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        self.model.eval()
        frozen_after = self.parameter_hash(False)
        adapter_after = self.parameter_hash(True)
        delta = sum(float((p.detach().float().cpu() - initial[n]).square().sum())
                    for n, p in trainable) ** 0.5
        self.model.save_pretrained(str(adapter_dir), safe_serialization=True)
        self.adapter_hash = adapter_after
        report = {"config": {"r": 4, "alpha": 8, "dropout": 0,
            "target_modules": ["q_proj", "v_proj"], "learning_rate": 2e-4,
            "steps": 12, "seed": 42, "max_sequence": 256, "microbatch": 1,
            "loss": "assistant-output-only cross entropy"},
            "eligible_examples": len(examples), "accepted_sequence_examples": len(data),
            "used_examples": len({s["task_id"] for s in steps}),
            "rejected_long": rejected, "eligible_pool_task_ids": [d["task_id"] for d in data],
            "training_task_ids": list(dict.fromkeys(s["task_id"] for s in steps)),
            "actual_optimized_ids": [s["task_id"] for s in steps],
            "trainable_parameters": sum(p.numel() for _, p in trainable),
            "trainable_names": [n for n, _ in trainable],
            "frozen_before": frozen_before, "frozen_after": frozen_after,
            "frozen_unchanged": frozen_before == frozen_after,
            "adapter_before": adapter_before, "adapter_after": adapter_after,
            "adapter_changed": adapter_before != adapter_after, "adapter_delta_l2": delta,
            "adapter_files": {p.name: file_digest(p) for p in Path(adapter_dir).iterdir()
                              if p.name == "adapter_config.json" or p.suffix == ".safetensors"},
            "losses": [s["loss"] for s in steps],
            "elapsed_s": time.perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        if not report["frozen_unchanged"] or not report["adapter_changed"] or delta <= 0:
            raise AssertionError("Real isolated parameter update verification failed")
        write_json(self.output / "training.json", report)
        return report

    def verify_reload(self, messages, adapter_dir):
        """Two real forwards, never a cached response as serialization evidence."""
        from copy import deepcopy
        from peft import get_peft_model_state_dict

        def state_hash(name):
            state = get_peft_model_state_dict(self.model, adapter_name=name)
            h = hashlib.sha256()
            for key, tensor in sorted(state.items()):
                h.update(key.encode())
                h.update(tensor.detach().contiguous().view(self.torch.uint8).cpu().numpy().tobytes())
            return h.hexdigest()

        original_state = state_hash("default")
        before = self.generate(messages, adapter=True, bypass_cache=True)
        old_hash = self.parameter_hash(True)
        # PEFT 0.18 creates additional adapters in backbone bf16 first. Loading
        # fp32 tensors into those would round them before its later fp32 cast.
        # Allocate and promote the destination BEFORE reading the saved weights.
        self.model.add_adapter("reloaded", deepcopy(self.model.peft_config["default"]))
        for name, parameter in self.model.named_parameters():
            if "lora_" in name and ".reloaded." in name:
                parameter.data = parameter.data.float()
        self.model.load_adapter(str(adapter_dir), adapter_name="reloaded", is_trainable=False)
        self.model.set_adapter("reloaded")
        restored_state = state_hash("reloaded")
        after = self.generate(messages, adapter=True, bypass_cache=True)
        equivalent = (before["generated_ids"] == after["generated_ids"]
                      and original_state == restored_state)
        report = {"pass": equivalent, "probe_key_before": before["key"], "probe_key_after": after["key"],
                  "adapter_training_hash": old_hash,
                  "same_greedy_tokens": before["generated_ids"] == after["generated_ids"],
                  "original_state_hash": original_state, "restored_state_hash": restored_state,
                  "exact_tensor_roundtrip": original_state == restored_state,
                  "before": before["raw"], "after": after["raw"], "actual_forwards": 2}
        write_json(self.output / "reload_check.json", report)
        if not equivalent:
            raise AssertionError("Adapter reload changed fixed-probe greedy tokens")
        return report
