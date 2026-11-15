#!/usr/bin/env bash
# Stop only this checkout's verified vLLM servers in the selected log directory.
set -euo pipefail

code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_dir=$(dirname -- "$code_dir")
log_dir=${CONFAGENT_SERVICE_LOG_DIR:-"$project_dir/logs"}
[[ -d "$log_dir" ]] || exit 0

"${CONFAGENT_PYTHON:-python3}" - "$project_dir" "$log_dir" "$@" <<'PY'
import errno
import os
from pathlib import Path
import re
import signal
import sys
import time

root = Path(sys.argv[1])
log_dir = Path(sys.argv[2]).resolve()
gpus = sys.argv[3:]
if any(not re.fullmatch(r"0|[1-9][0-9]*", gpu) for gpu in gpus):
    raise SystemExit("Supply GPU indices as separate arguments.")

run_filter = os.environ.get("CONFAGENT_STOP_RUN_ID")
targets = []
for path in sorted(log_dir.glob("confagent_gpu*_port*.pid")):
    match = re.fullmatch(r"confagent_gpu([0-9]+)_port([0-9]+)\.pid", path.name)
    if not match:
        continue
    gpu, port = match.groups()
    if gpus and gpu not in gpus:
        continue
    try:
        pid_text, run_id = path.read_text().split()
        pid = int(pid_text)
    except (ValueError, OSError):
        continue
    if pid < 2 or not re.fullmatch(r"[a-f0-9]{32}", run_id) or (run_filter and run_id != run_filter):
        continue
    targets.append((pid, gpu, port, run_id))


def owned(target):
    pid, gpu, port, run_id = target
    proc = Path("/proc") / str(pid)
    try:
        if proc.stat().st_uid != os.getuid():
            return False
        args = proc.joinpath("cmdline").read_bytes().split(b"\0")
        environment = set(proc.joinpath("environ").read_bytes().split(b"\0"))
        return args[1:4] == [b"-m", b"vllm.entrypoints.cli.main", b"serve"] and (
            b"--port" in args and args[args.index(b"--port") + 1] == port.encode()
        ) and {
            f"CUDA_VISIBLE_DEVICES={gpu}".encode(),
            f"CONFAGENT_SERVICE_ROOT={root}".encode(),
            f"CONFAGENT_SERVICE_LOG_DIR={log_dir}".encode(),
            f"CONFAGENT_SERVICE_RUN_ID={run_id}".encode(),
        }.issubset(environment)
    except (OSError, IndexError):
        return False


def stop(target):
    pid, _, _, _ = target
    if not owned(target):
        return False
    fd = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            try:
                fd = os.pidfd_open(pid)
            except OSError as exc:
                if exc.errno != errno.ENOSYS:
                    raise
        if not owned(target):
            return False
        if fd is None:
            os.kill(pid, signal.SIGTERM)
        else:
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return False
    finally:
        if fd is not None:
            os.close(fd)


stopping = {target for target in targets if stop(target)}
deadline = time.monotonic() + 30
while stopping and time.monotonic() < deadline:
    for target in tuple(stopping):
        if not owned(target):
            pid, gpu, _, _ = target
            print(f"Stopped project server {pid} on GPU {gpu}.")
            stopping.remove(target)
    if stopping:
        time.sleep(0.2)
if stopping:
    raise SystemExit("One or more project servers are still stopping.")
PY
