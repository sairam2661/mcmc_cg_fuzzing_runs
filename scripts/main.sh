#!/usr/bin/env bash

MODE="${1:-all}"
if [[ ! $MODE =~ ^(xml|sql|all)$ ]]; then
  echo "Usage: $0 [xml|sql|all]" >&2
  exit 1
fi

# —— load your experiments list ——
source "$(dirname "$0")/experiments.conf"

# —— Default env vars (override before calling if you like) ——
# : "${DATA_DIR}"
: "${OUTPUT_ROOT:=benchmarks/experiments}"
: "${HARNESS_DIR:=$DATA_DIR/harness}"
: "${HARNESS_COV_DIR:=$DATA_DIR/harness_cov}"

: "${AFL_TIMEOUT:=3600}"
: "${NUM_TRIALS:=5}"
: "${RANDOM_SEED_BASE:=42}"
: "${TRACK_INTERVAL:=30}"
: "${AFL_MAP_SIZE:=67133}"

: "${XML_EXP_NAME:=mcmc_restart_15}"
: "${SQL_EXP_NAME:=mcmc_prefix_10}"

export DATA_DIR OUTPUT_ROOT HARNESS_DIR HARNESS_COV_DIR
export AFL_TIMEOUT NUM_TRIALS RANDOM_SEED_BASE TRACK_INTERVAL AFL_MAP_SIZE
export XML_EXP_NAME SQL_EXP_NAME

# —— Step 1: Dependencies ——
echo "==> Installing dependencies"
scripts/setup_dependencies.sh

# —— Step 2: Build fuzzing targets ——
echo "==> Setting up fuzzing builds ($MODE)"
scripts/setup_fuzzing.sh "$MODE"

# —— Step 3: Build coverage harnesses ——
echo "==> Setting up coverage harnesses ($MODE)"
scripts/setup_harness.sh "$MODE"

if [[ $MODE == xml || $MODE == all ]]; then
  for exp in "${XML_EXPS[@]}"; do
    export XML_EXP_NAME="$exp"
    echo "--> XML experiment: $XML_EXP_NAME"
    scripts/run_fuzzing_experiment.sh xml
  done
fi

if [[ $MODE == sql || $MODE == all ]]; then
  for exp in "${SQL_EXPS[@]}"; do
    export SQL_EXP_NAME="$exp"
    echo "--> SQL experiment: $SQL_EXP_NAME"
    scripts/run_fuzzing_experiment.sh sql
  done
fi

echo "==> All done!"
