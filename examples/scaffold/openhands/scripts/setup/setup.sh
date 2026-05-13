#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Compatibility wrapper. Prefer the repository-level setup helper so benchmark
# environments stay isolated from one another.
exec bash "${REPO_ROOT}/examples/scripts/setup_benchmark_env.sh" swebench
