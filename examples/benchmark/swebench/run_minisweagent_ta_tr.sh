#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ROUTER_MODE="${ROUTER_MODE:-tr}"
export RUN_NAME="${RUN_NAME:-minisweagent_swebench_ta_tr_$(date +%Y%m%d_%H%M%S)}"
exec bash "${REPO_ROOT}/examples/benchmark/swebench/run_minisweagent_ta.sh" "$@"
