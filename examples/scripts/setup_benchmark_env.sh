#!/usr/bin/env bash
set -euo pipefail

# Create an isolated uv environment for one benchmark scaffold.
#
# Usage:
#   bash examples/scripts/setup_benchmark_env.sh swebench
#   bash examples/scripts/setup_benchmark_env.sh webarena
#   bash examples/scripts/setup_benchmark_env.sh osworld
#
# Override the destination with ENV_DIR=/path/to/venv.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCHMARK="${1:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-thunderagent}"
INSTALL_PLAYWRIGHT="${INSTALL_PLAYWRIGHT:-1}"

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage:
  bash examples/scripts/setup_benchmark_env.sh swebench
  bash examples/scripts/setup_benchmark_env.sh webarena
  bash examples/scripts/setup_benchmark_env.sh osworld

Environment overrides:
  ENV_DIR=/custom/env/path
  PYTHON_VERSION=3.12
  UV_CACHE_DIR=/tmp/uv-cache-thunderagent
  INSTALL_PLAYWRIGHT=0   # skip browser download for webarena
  OSWORLD_SOURCE_ROOT=/path/to/OSWorld
EOF
}

require_uv() {
  command -v uv >/dev/null 2>&1 || die "uv is required but was not found in PATH"
}

env_dir_for() {
  case "$1" in
    swebench|openhands) printf '%s/.venv-openhands' "${REPO_ROOT}" ;;
    webarena|agentlab-webarena) printf '%s/.venv-webarena' "${REPO_ROOT}" ;;
    osworld|osworld-verified) printf '%s/.venv-osworld' "${REPO_ROOT}" ;;
    *) return 1 ;;
  esac
}

install_base() {
  local env_dir="$1"
  export UV_CACHE_DIR
  log "Create/update env: ${env_dir}"
  uv venv "${env_dir}" --python "${PYTHON_VERSION}"
  log "Install ThunderAgent editable package"
  uv pip install --python "${env_dir}/bin/python" -e "${REPO_ROOT}"
}

install_openhands() {
  local env_dir="$1"
  install_base "${env_dir}"
  log "Install OpenHands scaffold dependencies"
  uv pip install --python "${env_dir}/bin/python" -e "${REPO_ROOT}/examples/scaffold/openhands"
  uv pip install --python "${env_dir}/bin/python" huggingface_hub
  log "OpenHands env ready: ${env_dir}"
}

install_webarena() {
  local env_dir="$1"
  install_base "${env_dir}"
  log "Install AgentLab/WebArena scaffold dependencies"
  uv pip install --python "${env_dir}/bin/python" -r "${REPO_ROOT}/examples/scaffold/agentlab_webarena/requirements.txt"
  log "Install NLTK tokenizer data for WebArena"
  mkdir -p "${env_dir}/nltk_data"
  NLTK_DATA="${env_dir}/nltk_data" "${env_dir}/bin/python" - <<'PY'
import nltk

nltk.download("punkt_tab", quiet=False, raise_on_error=True)
PY
  if [[ "${INSTALL_PLAYWRIGHT}" == "1" ]]; then
    log "Install Playwright Chromium browser"
    "${env_dir}/bin/python" -m playwright install chromium
  fi
  log "WebArena env ready: ${env_dir}"
}

install_osworld() {
  local env_dir="$1"
  install_base "${env_dir}"
  log "Install OSWorld scaffold dependencies"
  uv pip install --python "${env_dir}/bin/python" -r "${REPO_ROOT}/examples/scaffold/osworld/requirements.txt"
  if [[ -n "${OSWORLD_SOURCE_ROOT:-}" && -f "${OSWORLD_SOURCE_ROOT}/requirements.txt" ]]; then
    log "Install official OSWorld dependencies from ${OSWORLD_SOURCE_ROOT}"
    uv pip install --python "${env_dir}/bin/python" -r "${OSWORLD_SOURCE_ROOT}/requirements.txt"
  else
    log "Skip official OSWorld requirements: set OSWORLD_SOURCE_ROOT to an OSWorld checkout with requirements.txt"
  fi
  log "OSWorld env ready: ${env_dir}"
}

main() {
  [[ -n "${BENCHMARK}" ]] || { usage; exit 2; }
  require_uv

  local default_env_dir
  default_env_dir="$(env_dir_for "${BENCHMARK}")" || { usage; exit 2; }
  local env_dir="${ENV_DIR:-${default_env_dir}}"

  case "${BENCHMARK}" in
    swebench|openhands) install_openhands "${env_dir}" ;;
    webarena|agentlab-webarena) install_webarena "${env_dir}" ;;
    osworld|osworld-verified) install_osworld "${env_dir}" ;;
  esac

  log "Use this Python with runners:"
  log "  PYTHON_BIN=${env_dir}/bin/python"
}

main "$@"
