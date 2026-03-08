#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${HOME}/SphericalSBPOperators.jl"

JULIA_BIN="${JULIA_BIN:-${HOME}/.julia/juliaup/julia-1.13.0-beta2+0.x64.linux.gnu/bin/julia}"
RUN_ROOT="${RUN_ROOT:-remote_results/sbp8_vsearch_2026-03-08/exact_morphology/exact_${SLURM_JOB_ID}}"

mkdir -p "${RUN_ROOT}/results" "${RUN_ROOT}/logs"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"

export SBP8_RESULTS_DIR="${RUN_ROOT}/results"
export SBP8_OUT_BASENAME="${SBP8_OUT_BASENAME:-sbp8_exact_morphology}"
export SBP8_PAIR_FILE="${SBP8_PAIR_FILE:-remote_results/sbp8_vsearch_2026-03-07/distill_dense_20260307_194005/main/results/sbp8_distill_dense_pairs_gt_thresh.tsv}"
export SBP8_FR7="${SBP8_FR7:-11}"
export SBP8_BLOCK_STOP="${SBP8_BLOCK_STOP:-16}"
export SBP8_DIAG_FREE_END="${SBP8_DIAG_FREE_END:-24}"
export SBP8_TAIL_START="${SBP8_TAIL_START:-24}"
export SBP8_FIX_S1="${SBP8_FIX_S1:-true}"
export SBP8_S1_ANCHOR="${SBP8_S1_ANCHOR:-11/20}"
export SBP8_V11_FIX="${SBP8_V11_FIX:-1/1}"
export SBP8_ALPHA_STEPS="${SBP8_ALPHA_STEPS:-1/100,1/10,1/1}"
export SBP8_MAX_PASSES="${SBP8_MAX_PASSES:-40}"
export SBP8_STOP_ON_NEGATIVE="${SBP8_STOP_ON_NEGATIVE:-true}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "hostname=$(hostname)"
  echo "start_time=$(date -Is)"
  echo "julia_bin=${JULIA_BIN}"
  echo "run_root=${RUN_ROOT}"
  echo "sbp8_pair_file=${SBP8_PAIR_FILE}"
  echo "sbp8_fr7=${SBP8_FR7}"
  echo "sbp8_block_stop=${SBP8_BLOCK_STOP}"
  echo "sbp8_alpha_steps=${SBP8_ALPHA_STEPS}"
  echo "sbp8_max_passes=${SBP8_MAX_PASSES}"
} | tee "${RUN_ROOT}/logs/meta.txt"

"${JULIA_BIN}" --startup-file=no --project=. scripts/sbp8_exact_rational_morphology_search.jl \
  | tee "${RUN_ROOT}/logs/run.log"

echo "end_time=$(date -Is)" | tee -a "${RUN_ROOT}/logs/meta.txt"
