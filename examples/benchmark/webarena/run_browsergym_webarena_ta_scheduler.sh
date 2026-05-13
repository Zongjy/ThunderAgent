#!/usr/bin/env bash
set -euo pipefail

# BrowserGym + WebArena-Verified with ThunderAgent capacity scheduling enabled.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROUTER_MODE="${ROUTER_MODE:-tr}"
exec bash "${SCRIPT_DIR}/run_browsergym_webarena_ta_default.sh" "$@"
