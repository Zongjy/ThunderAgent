#!/usr/bin/env bash
set -euo pipefail

# BrowserGym + WebArena-Verified + ThunderAgent(default router) + vLLM.
# Run from repo root:
#   bash examples/benchmark/webarena/run_browsergym_webarena_ta_default.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCAFFOLD_DIR="${REPO_ROOT}/examples/scaffold/browsergym_webarena"
WEBARENA_ENV_DIR="${WEBARENA_ENV_DIR:-${REPO_ROOT}/.venv-webarena}"
THUNDERAGENT_ENV_DIR="${THUNDERAGENT_ENV_DIR:-${WEBARENA_ENV_DIR}}"
NLTK_DATA="${NLTK_DATA:-${WEBARENA_ENV_DIR}/nltk_data}"
WEBARENA_ENV_FILE="${WEBARENA_ENV_FILE:-${REPO_ROOT}/examples/benchmark/webarena/webarena.env}"
VLLM_ROOT="${VLLM_ROOT:-${HOME}/vllm}"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-${VLLM_ROOT}/.venv}"
VLLM_WORKDIR="${VLLM_WORKDIR:-${VLLM_ROOT}}"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
AGENT_LLM="${AGENT_LLM:-${SERVED_MODEL_NAME}}"
VLLM_PORT="${VLLM_PORT:-4747}"
TA_PORT="${TA_PORT:-9000}"
ROUTER_MODE="${ROUTER_MODE:-default}"

WEBARENA_TASK_IDS="${WEBARENA_TASK_IDS:-}"
WEBARENA_TASK_START="${WEBARENA_TASK_START:-0}"
WEBARENA_NUM_TASKS="${WEBARENA_NUM_TASKS:-1}"
WEBARENA_SITE_FILTER="${WEBARENA_SITE_FILTER:-}"
WEBARENA_BENCHMARK="${WEBARENA_BENCHMARK:-webarena_verified}"
WEBARENA_BENCHMARK_MODULE="${WEBARENA_BENCHMARK_MODULE:-browsergym.webarena_verified}"
WEBARENA_ENV_PREFIX="${WEBARENA_ENV_PREFIX:-browsergym/webarena_verified}"
WEBARENA_MAX_STEPS="${WEBARENA_MAX_STEPS:-30}"
WEBARENA_SEED="${WEBARENA_SEED:-0}"
WEBARENA_HEADLESS="${WEBARENA_HEADLESS:-1}"
WEBARENA_OBSERVATION_MODE="${WEBARENA_OBSERVATION_MODE:-axtree}"
WEBARENA_OBSERVATION_MAX_CHARS="${WEBARENA_OBSERVATION_MAX_CHARS:-24000}"
WEBARENA_COMPACT_ACTIONS="${WEBARENA_COMPACT_ACTIONS:-0}"

START_VLLM="${START_VLLM:-1}"
START_THUNDERAGENT="${START_THUNDERAGENT:-1}"
KEEP_SERVICES="${KEEP_SERVICES:-0}"
SAMPLE_METRICS="${SAMPLE_METRICS:-1}"
METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-5}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"

PYTHON_BIN="${PYTHON_BIN:-${THUNDERAGENT_ENV_DIR}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ENV_DIR}/bin/vllm}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_VLLM_USAGE_FLAGS="${ENABLE_VLLM_USAGE_FLAGS:-1}"
ENABLE_VLLM_TOOL_CALLING="${ENABLE_VLLM_TOOL_CALLING:-1}"
ENABLE_VLLM_KV_METRICS="${ENABLE_VLLM_KV_METRICS:-1}"
ENABLE_VLLM_LANGUAGE_MODEL_ONLY="${ENABLE_VLLM_LANGUAGE_MODEL_ONLY:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

RUN_NAME="${RUN_NAME:-browsergym_webarena_verified_${ROUTER_MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/examples/benchmark/webarena/runs}"

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
PROFILE_DIR="${RUN_DIR}/thunderagent_profiles"
MANIFEST="${RUN_DIR}/manifest.env"
RESULTS_DIR="${RUN_DIR}/browsergym_outputs"
SUMMARY_DIR="${RUN_DIR}/_analysis"

VLLM_LOG="${LOG_DIR}/vllm_${VLLM_PORT}.log"
TA_LOG="${LOG_DIR}/thunderagent_${TA_PORT}_${ROUTER_MODE}.log"
WEBARENA_LOG="${LOG_DIR}/browsergym_webarena.log"

VLLM_PID=""
TA_PID=""
SAMPLER_PID=""

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

load_webarena_env_file() {
  if [[ -f "${WEBARENA_ENV_FILE}" ]]; then
    log "Load WebArena env file: ${WEBARENA_ENV_FILE}"
    set -a
    # shellcheck disable=SC1090
    source "${WEBARENA_ENV_FILE}"
    set +a
  fi
}

resolve_task_ids() {
  if [[ -n "${WEBARENA_TASK_IDS}" ]]; then
    return
  fi
  [[ "${WEBARENA_NUM_TASKS}" =~ ^[0-9]+$ ]] || die "WEBARENA_NUM_TASKS must be an integer"
  [[ "${WEBARENA_TASK_START}" =~ ^[0-9]+$ ]] || die "WEBARENA_TASK_START must be an integer"
  (( WEBARENA_NUM_TASKS > 0 )) || die "Set WEBARENA_TASK_IDS or WEBARENA_NUM_TASKS>0"

  if [[ -n "${WEBARENA_SITE_FILTER}" ]]; then
    WEBARENA_TASK_IDS="$("${PYTHON_BIN}" - "${WEBARENA_SITE_FILTER}" "${WEBARENA_TASK_START}" "${WEBARENA_NUM_TASKS}" <<'PY'
import importlib.resources
import json
import sys

sites = {item.strip() for item in sys.argv[1].split(",") if item.strip()}
start = int(sys.argv[2])
limit = int(sys.argv[3])

data = json.loads(
    importlib.resources.files("webarena_verified")
    .joinpath("assets/dataset/webarena-verified.json")
    .read_text()
)
matched = [
    str(item["task_id"])
    for item in data
    if set(item.get("sites", [])) == sites
]
selected = matched[start : start + limit]
if len(selected) < limit:
    raise SystemExit(
        f"Requested {limit} tasks for sites={sorted(sites)}, "
        f"but only {len(matched)} matching tasks exist."
    )
print(",".join(selected))
PY
)"
    return
  fi

  local ids=()
  local end=$((WEBARENA_TASK_START + WEBARENA_NUM_TASKS))
  local i
  for ((i=WEBARENA_TASK_START; i<end; i++)); do
    ids+=("${i}")
  done
  local IFS=,
  WEBARENA_TASK_IDS="${ids[*]}"
}

check_webarena_env() {
  local missing=()
  local key
  for key in WA_SHOPPING WA_SHOPPING_ADMIN WA_REDDIT WA_GITLAB WA_WIKIPEDIA WA_MAP WA_HOMEPAGE; do
    [[ -n "${!key:-}" ]] || missing+=("${key}")
  done
  if (( ${#missing[@]} > 0 )); then
    die "Missing WebArena site environment variables: ${missing[*]}. Set them before running BrowserGym/WebArena-Verified."
  fi
}

wait_url() {
  local url="$1" name="$2" start now
  start="$(date +%s)"
  until curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; do
    now="$(date +%s)"
    (( now - start < HEALTH_TIMEOUT_S )) || die "${name} health timeout: ${url}"
    sleep 3
  done
  log "${name} ready: ${url}"
}

ensure_port_free() {
  local port="$1" name="$2"
  if command -v ss >/dev/null 2>&1 && ss -ltn "( sport = :${port} )" | grep -q ":${port}"; then
    die "${name} port ${port} is in use; set START_${name}=0 or use another port"
  fi
}

cleanup() {
  [[ "${KEEP_SERVICES}" == "1" ]] && { log "KEEP_SERVICES=1; services left running."; return; }
  set +e
  [[ -n "${SAMPLER_PID}" ]] && kill "${SAMPLER_PID}" 2>/dev/null
  [[ -n "${TA_PID}" ]] && kill "${TA_PID}" 2>/dev/null
  [[ -n "${VLLM_PID}" ]] && kill "${VLLM_PID}" 2>/dev/null
}
trap cleanup EXIT

write_manifest() {
  mkdir -p "${LOG_DIR}" "${PROFILE_DIR}" "${RESULTS_DIR}" "${SUMMARY_DIR}"
  cat > "${MANIFEST}" <<EOF
MODEL=${MODEL}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
AGENT_LLM=${AGENT_LLM}
VLLM_PORT=${VLLM_PORT}
TA_PORT=${TA_PORT}
ROUTER_MODE=${ROUTER_MODE}
WEBARENA_TASK_IDS=${WEBARENA_TASK_IDS}
WEBARENA_TASK_START=${WEBARENA_TASK_START}
WEBARENA_NUM_TASKS=${WEBARENA_NUM_TASKS}
WEBARENA_SITE_FILTER=${WEBARENA_SITE_FILTER}
WEBARENA_BENCHMARK=${WEBARENA_BENCHMARK}
WEBARENA_BENCHMARK_MODULE=${WEBARENA_BENCHMARK_MODULE}
WEBARENA_ENV_PREFIX=${WEBARENA_ENV_PREFIX}
WEBARENA_MAX_STEPS=${WEBARENA_MAX_STEPS}
WEBARENA_SEED=${WEBARENA_SEED}
WEBARENA_HEADLESS=${WEBARENA_HEADLESS}
WEBARENA_OBSERVATION_MODE=${WEBARENA_OBSERVATION_MODE}
WEBARENA_OBSERVATION_MAX_CHARS=${WEBARENA_OBSERVATION_MAX_CHARS}
WEBARENA_COMPACT_ACTIONS=${WEBARENA_COMPACT_ACTIONS}
START_VLLM=${START_VLLM}
START_THUNDERAGENT=${START_THUNDERAGENT}
KEEP_SERVICES=${KEEP_SERVICES}
SAMPLE_METRICS=${SAMPLE_METRICS}
PYTHON_BIN=${PYTHON_BIN}
THUNDERAGENT_ENV_DIR=${THUNDERAGENT_ENV_DIR}
WEBARENA_ENV_DIR=${WEBARENA_ENV_DIR}
WEBARENA_ENV_FILE=${WEBARENA_ENV_FILE}
NLTK_DATA=${NLTK_DATA}
VLLM_ROOT=${VLLM_ROOT}
VLLM_ENV_DIR=${VLLM_ENV_DIR}
VLLM_WORKDIR=${VLLM_WORKDIR}
VLLM_BIN=${VLLM_BIN}
HF_ENDPOINT=${HF_ENDPOINT}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}
ENABLE_VLLM_USAGE_FLAGS=${ENABLE_VLLM_USAGE_FLAGS}
ENABLE_VLLM_TOOL_CALLING=${ENABLE_VLLM_TOOL_CALLING}
ENABLE_VLLM_KV_METRICS=${ENABLE_VLLM_KV_METRICS}
ENABLE_VLLM_LANGUAGE_MODEL_ONLY=${ENABLE_VLLM_LANGUAGE_MODEL_ONLY}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER}
REASONING_PARSER=${REASONING_PARSER}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}
RUN_NAME=${RUN_NAME}
OUTPUT_ROOT=${OUTPUT_ROOT}
RUN_DIR=${RUN_DIR}
EOF
}

start_vllm() {
  [[ "${START_VLLM}" == "1" ]] || { log "Reuse vLLM: ${VLLM_PORT}"; wait_url "http://127.0.0.1:${VLLM_PORT}/health" "vLLM"; return; }
  ensure_port_free "${VLLM_PORT}" "VLLM"

  local cmd=(
    "${VLLM_BIN}" serve "${MODEL}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${VLLM_PORT}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
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
  nohup env VIRTUAL_ENV="${VLLM_ENV_DIR}" PATH="${VLLM_ENV_DIR}/bin:${PATH}" HF_ENDPOINT="${HF_ENDPOINT}" "${cmd[@]}" > "${VLLM_LOG}" 2>&1 &
  VLLM_PID="$!"
  popd >/dev/null
  wait_url "http://127.0.0.1:${VLLM_PORT}/health" "vLLM"
}

start_thunderagent() {
  [[ "${START_THUNDERAGENT}" == "1" ]] || { log "Reuse ThunderAgent: ${TA_PORT}"; wait_url "http://127.0.0.1:${TA_PORT}/health" "ThunderAgent"; return; }
  ensure_port_free "${TA_PORT}" "THUNDERAGENT"

  log "Start ThunderAgent router=${ROUTER_MODE}: ${TA_LOG}"
  nohup env VIRTUAL_ENV="${THUNDERAGENT_ENV_DIR}" PATH="${THUNDERAGENT_ENV_DIR}/bin:${PATH}" PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m ThunderAgent \
    --backend-type vllm \
    --backends "http://127.0.0.1:${VLLM_PORT}" \
    --port "${TA_PORT}" \
    --router "${ROUTER_MODE}" \
    --metrics \
    --profile \
    --profile-dir "${PROFILE_DIR}" \
    > "${TA_LOG}" 2>&1 &
  TA_PID="$!"
  wait_url "http://127.0.0.1:${TA_PORT}/health" "ThunderAgent"
}

start_sampler() {
  [[ "${SAMPLE_METRICS}" == "1" ]] || return
  (
    while true; do
      printf '{"sample_ts":%s,"payload":%s}\n' "$(date +%s)" "$(curl -fsS --max-time 5 "http://127.0.0.1:${TA_PORT}/health" 2>/dev/null || echo '{}')" >> "${LOG_DIR}/thunderagent_health.jsonl"
      printf '{"sample_ts":%s,"payload":%s}\n' "$(date +%s)" "$(curl -fsS --max-time 5 "http://127.0.0.1:${TA_PORT}/metrics" 2>/dev/null || echo '{}')" >> "${LOG_DIR}/thunderagent_metrics.jsonl"
      { printf '# sample_ts=%s\n' "$(date +%s)"; curl -fsS --max-time 5 "http://127.0.0.1:${VLLM_PORT}/metrics" 2>/dev/null || true; printf '\n'; } >> "${LOG_DIR}/vllm_metrics.prom"
      sleep "${METRICS_INTERVAL_S}"
    done
  ) &
  SAMPLER_PID="$!"
}

run_webarena() {
  local args=(
    "${SCAFFOLD_DIR}/run_browsergym_webarena.py"
    --base-url "http://127.0.0.1:${TA_PORT}/v1"
    --model "${AGENT_LLM}"
    --benchmark "${WEBARENA_BENCHMARK}"
    --benchmark-module "${WEBARENA_BENCHMARK_MODULE}"
    --task-ids "${WEBARENA_TASK_IDS}"
    --env-prefix "${WEBARENA_ENV_PREFIX}"
    --seed "${WEBARENA_SEED}"
    --max-steps "${WEBARENA_MAX_STEPS}"
    --observation-mode "${WEBARENA_OBSERVATION_MODE}"
    --observation-max-chars "${WEBARENA_OBSERVATION_MAX_CHARS}"
    --output-dir "${RESULTS_DIR}"
  )
  [[ "${WEBARENA_HEADLESS}" == "1" ]] && args+=(--headless) || args+=(--no-headless)
  [[ "${WEBARENA_COMPACT_ACTIONS}" == "1" ]] && args+=(--compact-actions)

  log "Run BrowserGym ${WEBARENA_BENCHMARK}: tasks=${WEBARENA_TASK_IDS}, max_steps=${WEBARENA_MAX_STEPS}"
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" VIRTUAL_ENV="${THUNDERAGENT_ENV_DIR}" PATH="${THUNDERAGENT_ENV_DIR}/bin:${PATH}" NLTK_DATA="${NLTK_DATA}" "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${WEBARENA_LOG}"
}

main() {
  [[ -d "${SCAFFOLD_DIR}" ]] || die "BrowserGym scaffold not found: ${SCAFFOLD_DIR}"
  command -v curl >/dev/null 2>&1 || die "Missing command: curl"
  command -v tee >/dev/null 2>&1 || die "Missing command: tee"
  [[ -x "${PYTHON_BIN}" ]] || command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Cannot find Python: ${PYTHON_BIN}. Run: bash examples/scripts/setup_benchmark_env.sh webarena"
  [[ "${START_VLLM}" != "1" ]] || [[ -d "${VLLM_WORKDIR}" ]] || die "vLLM workdir not found: ${VLLM_WORKDIR}"

  load_webarena_env_file
  resolve_task_ids
  check_webarena_env
  write_manifest
  log "Run dir: ${RUN_DIR}"
  start_vllm
  start_thunderagent
  start_sampler
  run_webarena
  "${PYTHON_BIN}" "${REPO_ROOT}/examples/benchmark/webarena/analyze_webarena_outputs.py" --root "${RESULTS_DIR}" --output-dir "${SUMMARY_DIR}"
  log "Done. Results: ${RESULTS_DIR}/results.json"
  log "Summary: ${SUMMARY_DIR}/webarena_analysis.json"
  log "Profile CSV: ${PROFILE_DIR}/step_profiles.csv"
}

main "$@"
