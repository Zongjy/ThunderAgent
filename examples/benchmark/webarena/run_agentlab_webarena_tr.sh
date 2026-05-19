#!/usr/bin/env bash
set -euo pipefail

# AgentLab + WebArena through ThunderAgent with the TR scheduler.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export ROUTER_MODE="${ROUTER_MODE:-tr}"
export RUN_NAME="${RUN_NAME:-agentlab_webarena_tr_$(date +%Y%m%d_%H%M%S)}"

exec bash "${REPO_ROOT}/examples/benchmark/webarena/run_agentlab_webarena_ta.sh" "$@"

