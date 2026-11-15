#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname -- "$0")/launch_replicas.sh" "${GPU:-0}" "${PORT:-18000}"
