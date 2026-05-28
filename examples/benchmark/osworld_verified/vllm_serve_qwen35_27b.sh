#!/usr/bin/env bash
set -euo pipefail

# Start Qwen3.5-27B with vLLM for ThunderAgent OSWorld experiments.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VLLM_ROOT="${VLLM_ROOT:-${HOME}/vllm}"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-${VLLM_ROOT}/.venv}"
VLLM_WORKDIR="${VLLM_WORKDIR:-${VLLM_ROOT}}"

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
if [[ -z "${SERVED_MODEL_NAME:-}" ]]; then
  if [[ "${MODEL}" == "Qwen/Qwen3.5-27B" ]]; then
    SERVED_MODEL_NAME="qwen3.5-27B"
  else
    SERVED_MODEL_NAME="${MODEL}"
  fi
fi

VLLM_BIN="${VLLM_BIN:-${VLLM_ENV_DIR}/bin/vllm}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_HEALTH_HOST="${VLLM_HEALTH_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-4747}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://${VLLM_HEALTH_HOST}:${VLLM_PORT}/health}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_MM_ENCODER_TP_MODE="${VLLM_MM_ENCODER_TP_MODE:-data}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_VLLM_USAGE_FLAGS="${ENABLE_VLLM_USAGE_FLAGS:-1}"
ENABLE_VLLM_TOOL_CALLING="${ENABLE_VLLM_TOOL_CALLING:-1}"
ENABLE_VLLM_KV_METRICS="${ENABLE_VLLM_KV_METRICS:-1}"
ENABLE_VLLM_LANGUAGE_MODEL_ONLY="${ENABLE_VLLM_LANGUAGE_MODEL_ONLY:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
VLLM_ENFORCE_STRICT_TOOL_CALLING="${VLLM_ENFORCE_STRICT_TOOL_CALLING:-1}"
REASONING_PARSER="${REASONING_PARSER:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---trust-remote-code}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/examples/benchmark/osworld_verified/runs}"
VLLM_RUN_NAME="${VLLM_RUN_NAME:-vllm_qwen35_27b_${VLLM_PORT}_$(date +%Y%m%d_%H%M%S)}"
VLLM_RUN_DIR="${VLLM_RUN_DIR:-${OUTPUT_ROOT}/${VLLM_RUN_NAME}}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-${VLLM_RUN_DIR}/logs}"
VLLM_LOG="${VLLM_LOG:-${VLLM_LOG_DIR}/vllm_${VLLM_PORT}.log}"
VLLM_PID_FILE="${VLLM_PID_FILE:-${VLLM_RUN_DIR}/vllm_${VLLM_PORT}.pid}"
VLLM_MANIFEST="${VLLM_MANIFEST:-${VLLM_RUN_DIR}/manifest.env}"
VLLM_REUSE_IF_RUNNING="${VLLM_REUSE_IF_RUNNING:-1}"
VLLM_FOREGROUND="${VLLM_FOREGROUND:-0}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

wait_url() {
  local url="$1" name="$2" start now_ts
  start="$(date +%s)"
  until curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; do
    now_ts="$(date +%s)"
    (( now_ts - start < HEALTH_TIMEOUT_S )) || die "${name} health timeout: ${url}"
    sleep 3
  done
  log "${name} ready: ${url}"
}

port_is_listening() {
  command -v ss >/dev/null 2>&1 || return 1
  ss -ltn "( sport = :${VLLM_PORT} )" | grep -q ":${VLLM_PORT}"
}

write_manifest() {
  mkdir -p "${VLLM_LOG_DIR}" "$(dirname "${VLLM_PID_FILE}")"
  cat > "${VLLM_MANIFEST}" <<EOF
MODEL=${MODEL}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
VLLM_ROOT=${VLLM_ROOT}
VLLM_ENV_DIR=${VLLM_ENV_DIR}
VLLM_WORKDIR=${VLLM_WORKDIR}
VLLM_BIN=${VLLM_BIN}
VLLM_HOST=${VLLM_HOST}
VLLM_HEALTH_HOST=${VLLM_HEALTH_HOST}
VLLM_PORT=${VLLM_PORT}
VLLM_HEALTH_URL=${VLLM_HEALTH_URL}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}
VLLM_MM_ENCODER_TP_MODE=${VLLM_MM_ENCODER_TP_MODE}
HF_ENDPOINT=${HF_ENDPOINT}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}
ENABLE_VLLM_USAGE_FLAGS=${ENABLE_VLLM_USAGE_FLAGS}
ENABLE_VLLM_TOOL_CALLING=${ENABLE_VLLM_TOOL_CALLING}
ENABLE_VLLM_KV_METRICS=${ENABLE_VLLM_KV_METRICS}
ENABLE_VLLM_LANGUAGE_MODEL_ONLY=${ENABLE_VLLM_LANGUAGE_MODEL_ONLY}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER}
VLLM_ENFORCE_STRICT_TOOL_CALLING=${VLLM_ENFORCE_STRICT_TOOL_CALLING}
REASONING_PARSER=${REASONING_PARSER}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}
VLLM_RUN_NAME=${VLLM_RUN_NAME}
VLLM_RUN_DIR=${VLLM_RUN_DIR}
VLLM_LOG=${VLLM_LOG}
VLLM_PID_FILE=${VLLM_PID_FILE}
EOF
}

main() {
  command -v curl >/dev/null 2>&1 || die "Missing command: curl"
  [[ -d "${VLLM_WORKDIR}" ]] || die "vLLM workdir not found: ${VLLM_WORKDIR}"
  [[ -x "${VLLM_BIN}" ]] || command -v "${VLLM_BIN}" >/dev/null 2>&1 || die "Cannot find vLLM binary: ${VLLM_BIN}"

  write_manifest

  if curl -fsS --max-time 3 "${VLLM_HEALTH_URL}" >/dev/null 2>&1; then
    [[ "${VLLM_REUSE_IF_RUNNING}" == "1" ]] || die "vLLM is already healthy at ${VLLM_HEALTH_URL}"
    log "Reuse existing vLLM: ${VLLM_HEALTH_URL}"
    log "Manifest: ${VLLM_MANIFEST}"
    return 0
  fi
  if port_is_listening; then
    die "vLLM port ${VLLM_PORT} is in use, but health check failed: ${VLLM_HEALTH_URL}"
  fi

  local cmd=(
    "${VLLM_BIN}" serve "${MODEL}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --mm-encoder-tp-mode "${VLLM_MM_ENCODER_TP_MODE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
  )
  [[ "${ENABLE_PREFIX_CACHING}" == "1" ]] && cmd+=(--enable-prefix-caching)
  [[ -n "${REASONING_PARSER}" ]] && cmd+=(--reasoning-parser "${REASONING_PARSER}")
  [[ "${ENABLE_VLLM_LANGUAGE_MODEL_ONLY}" == "1" ]] && cmd+=(--language-model-only)
  [[ "${ENABLE_VLLM_USAGE_FLAGS}" == "1" ]] && cmd+=(--enable-prompt-tokens-details --enable-force-include-usage)
  [[ "${ENABLE_VLLM_TOOL_CALLING}" == "1" ]] && cmd+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
  [[ "${ENABLE_VLLM_KV_METRICS}" == "1" ]] && cmd+=(--enable-mfu-metrics --kv-cache-metrics)
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    read -r -a extra_args <<< "${VLLM_EXTRA_ARGS}"
    cmd+=("${extra_args[@]}")
  fi

  log "Start vLLM from ${VLLM_WORKDIR}: ${VLLM_LOG}"
  pushd "${VLLM_WORKDIR}" >/dev/null
  if [[ "${VLLM_FOREGROUND}" == "1" ]]; then
    exec env VIRTUAL_ENV="${VLLM_ENV_DIR}" PATH="${VLLM_ENV_DIR}/bin:${PATH}" HF_ENDPOINT="${HF_ENDPOINT}" VLLM_ENFORCE_STRICT_TOOL_CALLING="${VLLM_ENFORCE_STRICT_TOOL_CALLING}" "${cmd[@]}"
  fi
  nohup env VIRTUAL_ENV="${VLLM_ENV_DIR}" PATH="${VLLM_ENV_DIR}/bin:${PATH}" HF_ENDPOINT="${HF_ENDPOINT}" VLLM_ENFORCE_STRICT_TOOL_CALLING="${VLLM_ENFORCE_STRICT_TOOL_CALLING}" "${cmd[@]}" > "${VLLM_LOG}" 2>&1 &
  local vllm_pid="$!"
  popd >/dev/null
  echo "${vllm_pid}" > "${VLLM_PID_FILE}"
  wait_url "${VLLM_HEALTH_URL}" "vLLM"
  log "PID: ${vllm_pid}"
  log "Run dir: ${VLLM_RUN_DIR}"
}

main "$@"
