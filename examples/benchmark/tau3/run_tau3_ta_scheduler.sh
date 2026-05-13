#!/usr/bin/env bash
set -euo pipefail

# Official tau2/τ³ benchmark with ThunderAgent capacity scheduling enabled.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROUTER_MODE="${ROUTER_MODE:-tr}"
exec bash "${SCRIPT_DIR}/run_tau3_ta_default.sh" "$@"
