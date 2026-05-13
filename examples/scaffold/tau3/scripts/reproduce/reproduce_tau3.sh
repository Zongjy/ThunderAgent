#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

exec bash "${REPO_ROOT}/examples/benchmark/tau3/run_tau3_ta_default.sh" "$@"
