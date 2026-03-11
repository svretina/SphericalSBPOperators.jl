#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${HOME}/SphericalSBPOperators.jl"

JULIA_BIN="${JULIA_BIN:-${HOME}/.julia/juliaup/julia-1.13.0-beta2+0.x64.linux.gnu/bin/julia}"
RUN_ROOT="${RUN_ROOT:-remote_results/sbp8_vsearch_2026-03-07/strategy_jobs/strategy_${SLURM_JOB_ID}}"

mkdir -p "${RUN_ROOT}/results" "${RUN_ROOT}/logs"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"

export SBP8_RESULTS_DIR="${RUN_ROOT}/results"
export SBP8_OUT_BASENAME="${SBP8_OUT_BASENAME:-sbp8_structure_strategy}"
export SBP8_STRATEGY="${SBP8_STRATEGY:-dense_nullspace}"
export SBP8_FR7="${SBP8_FR7:-11}"
export SBP8_BLOCK_STOP="${SBP8_BLOCK_STOP:-16}"
export SBP8_DIAG_FREE_END="${SBP8_DIAG_FREE_END:-24}"
export SBP8_TAIL_START="${SBP8_TAIL_START:-24}"
export SBP8_FIX_S1="${SBP8_FIX_S1:-true}"
export SBP8_S1_ANCHOR="${SBP8_S1_ANCHOR:-11/20}"
export SBP8_V11_FIX="${SBP8_V11_FIX:-1/1}"
export SBP8_STOP_ON_STRICT="${SBP8_STOP_ON_STRICT:-false}"
export SBP8_DENSE_MAX_COLS="${SBP8_DENSE_MAX_COLS:-24}"
export SBP8_DENSE_MAX_PASSES="${SBP8_DENSE_MAX_PASSES:-6}"
export SBP8_DENSE_DENS="${SBP8_DENSE_DENS:-1000,10000,100000}"
export SBP8_GREEDY_ITERS="${SBP8_GREEDY_ITERS:-240}"
export SBP8_GREEDY_TEMP0="${SBP8_GREEDY_TEMP0:-0.25}"
export SBP8_GREEDY_COOL="${SBP8_GREEDY_COOL:-0.985}"
export SBP8_GREEDY_W_IMAG="${SBP8_GREEDY_W_IMAG:-4.0}"
export SBP8_GREEDY_SEED="${SBP8_GREEDY_SEED:-20260307}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "hostname=$(hostname)"
  echo "start_time=$(date -Is)"
  echo "julia_bin=${JULIA_BIN}"
  echo "run_root=${RUN_ROOT}"
  echo "sbp8_strategy=${SBP8_STRATEGY}"
  echo "sbp8_fr7=${SBP8_FR7}"
  echo "sbp8_block_stop=${SBP8_BLOCK_STOP}"
} | tee "${RUN_ROOT}/logs/meta.txt"

"${JULIA_BIN}" --startup-file=no --project=. scripts/sbp8_structure_strategy_search.jl \
  | tee "${RUN_ROOT}/logs/run.log"

echo "end_time=$(date -Is)" | tee -a "${RUN_ROOT}/logs/meta.txt"
