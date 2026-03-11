#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${HOME}/SphericalSBPOperators.jl"

JULIA_BIN="${JULIA_BIN:-${HOME}/.julia/juliaup/julia-1.13.0-beta2+0.x64.linux.gnu/bin/julia}"
RUN_ROOT="${RUN_ROOT:-remote_results/sbp8_vsearch_2026-03-06/slurm/sbp8_runB_${SLURM_JOB_ID}}"

mkdir -p "${RUN_ROOT}/results" "${RUN_ROOT}/logs"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"

export SBP8_RESULTS_DIR="${RUN_ROOT}/results"
export SBP8_SPEC_MODE="${SBP8_SPEC_MODE:-extended}"
export SBP8_BASE_ZERO_MODE="${SBP8_BASE_ZERO_MODE:-relaxed_1416_1516}"
export SBP8_FR7_LIST="${SBP8_FR7_LIST:-11}"
export SBP8_STAGE_SPAN="${SBP8_STAGE_SPAN:-4}"
export SBP8_BR="${SBP8_BR:-19}"
export SBP8_BW="${SBP8_BW:-7}"
export SBP8_DIAG_FREE_END="${SBP8_DIAG_FREE_END:-24}"
export SBP8_TAIL_START="${SBP8_TAIL_START:-24}"
export SBP8_S1_ANCHOR="${SBP8_S1_ANCHOR:-11/20}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "hostname=$(hostname)"
  echo "start_time=$(date -Is)"
  echo "julia_bin=${JULIA_BIN}"
  echo "run_root=${RUN_ROOT}"
  echo "spec_mode=${SBP8_SPEC_MODE}"
  echo "base_zero_mode=${SBP8_BASE_ZERO_MODE}"
  echo "fr7_list=${SBP8_FR7_LIST}"
} | tee "${RUN_ROOT}/logs/meta.txt"

"${JULIA_BIN}" --startup-file=no --project=. scripts/sbp8_runB_relax_parallel_search.jl \
  | tee "${RUN_ROOT}/logs/run.log"

echo "end_time=$(date -Is)" | tee -a "${RUN_ROOT}/logs/meta.txt"
