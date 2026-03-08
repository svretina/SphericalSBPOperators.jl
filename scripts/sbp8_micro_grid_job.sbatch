#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${HOME}/SphericalSBPOperators.jl"

JULIA_BIN="${JULIA_BIN:-${HOME}/.julia/juliaup/julia-1.13.0-beta2+0.x64.linux.gnu/bin/julia}"
RUN_ROOT="${RUN_ROOT:-remote_results/sbp8_vsearch_2026-03-08/microgrid/microgrid_${SLURM_JOB_ID}}"

mkdir -p "${RUN_ROOT}/results" "${RUN_ROOT}/logs"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"

export SBP8_RESULTS_DIR="${RUN_ROOT}/results"
export SBP8_OUT_BASENAME="${SBP8_OUT_BASENAME:-sbp8_exact_subspace_microgrid}"
export SBP8_PAIR_FILE="${SBP8_PAIR_FILE:-remote_results/sbp8_vsearch_2026-03-07/distill_dense_20260307_194005/main/results/sbp8_distill_dense_pairs_gt_thresh.tsv}"
# Use the best alpha coordinates from the previous 408931 Nelder-Mead run as the anchor
export SBP8_ALPHA0_FILE="${SBP8_ALPHA0_FILE:-remote_results/sbp8_vsearch_2026-03-08/subspace_opt/subspace_408931/results/sbp8_exact_subspace_opt_alpha_float.tsv}"
export SBP8_REQUIRE_ALPHA0_FILE="true"

export SBP8_FR7="${SBP8_FR7:-11}"
export SBP8_BLOCK_STOP="${SBP8_BLOCK_STOP:-16}"
export SBP8_DIAG_FREE_END="${SBP8_DIAG_FREE_END:-24}"
export SBP8_TAIL_START="${SBP8_TAIL_START:-24}"
export SBP8_FIX_S1="${SBP8_FIX_S1:-true}"
export SBP8_S1_ANCHOR="${SBP8_S1_ANCHOR:-11/20}"
export SBP8_V11_FIX="${SBP8_V11_FIX:-1/1}"

# Enable Microgrid direct terrace search
export SBP8_OPT_METHOD="microgrid"
export SBP8_OPT_PENALTY_IMAG="${SBP8_OPT_PENALTY_IMAG:-1000.0}"
# Change internal rounding tolerance to zoom into the 1e-9 terrace strictly!
export SBP8_EVAL_ALPHA_TOL="${SBP8_EVAL_ALPHA_TOL:-1e-9}"
export SBP8_TARGET_MAX_REAL="${SBP8_TARGET_MAX_REAL:--1e-8}"
export SBP8_FLOAT_IMAG_TOL="${SBP8_FLOAT_IMAG_TOL:-1e-12}"
export SBP8_RATIONAL_TOLS="${SBP8_RATIONAL_TOLS:-1e-8,1e-9,1e-10,1e-11}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "hostname=$(hostname)"
  echo "start_time=$(date -Is)"
  echo "julia_bin=${JULIA_BIN}"
  echo "run_root=${RUN_ROOT}"
  echo "sbp8_pair_file=${SBP8_PAIR_FILE}"
  echo "sbp8_alpha0_file=${SBP8_ALPHA0_FILE}"
  echo "sbp8_opt_method=${SBP8_OPT_METHOD}"
  echo "sbp8_eval_alpha_tol=${SBP8_EVAL_ALPHA_TOL}"
  echo "sbp8_rational_tols=${SBP8_RATIONAL_TOLS}"
} | tee "${RUN_ROOT}/logs/meta.txt"

"${JULIA_BIN}" --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()' \
  | tee "${RUN_ROOT}/logs/run.log"

"${JULIA_BIN}" --startup-file=no --project=. scripts/sbp8_exact_subspace_opt.jl \
  | tee -a "${RUN_ROOT}/logs/run.log"

echo "end_time=$(date -Is)" | tee -a "${RUN_ROOT}/logs/meta.txt"
