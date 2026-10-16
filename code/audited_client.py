"""Record actual model requests and optionally reuse identical deterministic ones."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from openai.types.chat import ChatCompletion


class AuditedClient:
    def __init__(self, client, path, resume=False, identity=None, before_request=None):
        self.client = client
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.cache_hits = 0
        self.actual_calls = 0
        self.identity = identity
        self.before_request = before_request
        identity_path = self.path.with_suffix(".identity.json")
        if identity is not None:
            if identity_path.exists():
                if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                    raise ValueError("Runtime identity changed; refusing cached responses")
            elif self.path.exists():
                raise ValueError("Cannot resume unidentified request log")
            else:
                identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if self.path.exists() and not resume:
            raise FileExistsError("Request log exists; use a new run directory or explicit resume")
        if resume and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if "response" in row:
                    self.cache[row["request_sha256"]] = row["response"]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        payload = dict(kwargs)
        # Do not let a warm server prompt cache change the local comparison.
        payload["extra_body"] = {**payload.get("extra_body", {}), "cache_prompt": False}
        cache_body = {"identity": self.identity, "request": payload} if self.identity is not None else payload
        key = hashlib.sha256(json.dumps(cache_body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in self.cache and payload.get("temperature", 0) == 0:
            self.cache_hits += 1
            return ChatCompletion.model_validate(self.cache[key])
        row = {"request_sha256": key, "request": payload}
        if self.before_request is not None:
            row["thermal"] = self.before_request()
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**payload)
            row["response"] = response.model_dump()
            self.cache[key] = row["response"]
            return response
        except Exception as exc:
            row["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            self.actual_calls += 1
            row["latency_s"] = time.perf_counter() - start
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def gpu_temperature():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10, check=True)
        return float(result.stdout.splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def wait_for_gpu(limit=70, read=gpu_temperature, sleep=time.sleep):
    waited, temperature = 0, read()
    while temperature is not None and temperature > limit:
        if waited >= 300:
            raise TimeoutError("GPU did not cool below the configured request threshold")
        sleep(2)
        waited += 2
        temperature = read()
    return {"temperature_before_c": temperature, "cooldown_s": waited, "request_threshold_c": limit}


def local_client(base_url, log_path, resume=False, identity=None):
    import httpx
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="local", max_retries=0, timeout=120,
                    http_client=httpx.Client(trust_env=False))
    return AuditedClient(client, log_path, resume, identity, before_request=wait_for_gpu)


def local_runtime_identity(base_url, runtime):
    """Bind a local run to actual model bytes, server binary and reported build."""
    import httpx
    runtime = Path(runtime).resolve()
    manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
    with httpx.Client(trust_env=False, timeout=10) as client:
        response = client.get(base_url.removesuffix("/v1").rstrip("/") + "/props")
        response.raise_for_status()
        props = response.json()
    model_path = Path(props["model_path"]).resolve()
    expected_model = runtime / "models" / "Qwen3.5-0.8B-Q4_K_M.gguf"
    if model_path != expected_model:
        raise ValueError("Service is not using the provisioned local 0.8B model")
    def file_hash(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()
    model_hash = file_hash(model_path)
    expected_hash = next(a["sha256"] for a in manifest["assets"] if a["relative_path"].startswith("models/"))
    if model_hash != expected_hash:
        raise ValueError("Provisioned model bytes changed")
    return {"endpoint": base_url, "model_file": model_path.name, "model_sha256": model_hash,
            "model_revision": manifest["model_revision"], "llama_tag": manifest["llama_tag"],
            "server_exe_sha256": file_hash(runtime / "llama/llama-server.exe"),
            "build_info": props.get("build_info"), "model_alias": props.get("model_alias"),
            "context": props.get("default_generation_settings", {}).get("n_ctx"),
            "slots": props.get("total_slots")}
