#!/usr/bin/env bash
set -euo pipefail

# OSWorld-Verified with Agent-S and ThunderAgent capacity scheduling.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ROUTER_MODE="${ROUTER_MODE:-tr}"

exec bash "${REPO_ROOT}/examples/benchmark/osworld_verified/run_osworld_verified_ta_default.sh" "$@"
