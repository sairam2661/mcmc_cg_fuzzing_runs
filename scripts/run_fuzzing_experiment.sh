#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ———————— 1. Global config ————————
DATA_DIR="${DATA_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmarks/experiments}"
TRACK_INTERVAL="${TRACK_INTERVAL:-30}"       # secs between coverage polls
AFL_TIMEOUT="${AFL_TIMEOUT:-3600}"        # secs per trial
NUM_TRIALS="${NUM_TRIALS:-3}"
RANDOM_SEED_BASE="${RANDOM_SEED_BASE:-42}"

export AFL_SKIP_CPUFREQ=1
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
export AFL_NO_UI=1

# SQL needs a larger map
export AFL_MAP_SIZE="${AFL_MAP_SIZE:-67133}"

# ———————— 2. Experiment‐specific config ————————

# XML
XML_EXP_NAME="${XML_EXP_NAME}"
XML_BASE="$OUTPUT_ROOT/xml/$XML_EXP_NAME"
XML_SEEDS_DIR="$XML_BASE/llm_seeds"
XML_HARNESS="${XML_HARNESS:-$DATA_DIR/harness/xmllint}"
XML_HARNESS_COV="${XML_HARNESS_COV:-$DATA_DIR/harness_cov/xmllint_cov}"


# SQL
SQL_EXP_NAME="${SQL_EXP_NAME}"
SQL_BASE="$OUTPUT_ROOT/sql/$SQL_EXP_NAME"
SQL_SEEDS_DIR="$SQL_BASE/llm_seeds"
SQL_HARNESS="${SQL_HARNESS:-$DATA_DIR/harness/sqlite_fuzz}"
SQL_HARNESS_COV="${SQL_HARNESS_COV:-$DATA_DIR/harness_cov/sqlite_fuzz_cov}"


# ———————— 3. Boilerplate functions ————————

setup_trial_dirs() {
  local base=$1 trial=$2
  local odir="$base/trial_$trial"
  mkdir -p "$odir/output/default/queue" "$odir/coverage"
  printf 'timestamp,experiment,line_coverage,branch_coverage,function_coverage,llm_seeds_count,total_corpus\n' \
    >"$odir/coverage/coverage_data.csv"
}

track_coverage() {
  local exp=$1 base=$2 trial=$3 seeds_ext=$4
  local trial_dir="$base/trial_$trial"
  local tsf=$(date +%Y%m%d_%H%M%S)
  local tmp="$trial_dir/coverage/tmp_$tsf"
  mkdir -p "$tmp"

  # 1) Count LLM seeds and AFL corpus
  local llm_count
  llm_count=$(find "$base/llm_seeds" -maxdepth 1 -type f -name "*.$seeds_ext" | wc -l)
  local queue="$trial_dir/output/default/queue"
  local corpus=0
  [ -d "$queue" ] && corpus=$(find "$queue" -name 'id:*' -type f | wc -l)

  # 2) Run coverage for every test
  for f in "$queue"/id:*; do
    [ -f "$f" ] && LLVM_PROFILE_FILE="$tmp/$(basename "$f").profraw" \
      "$HARNESS_COV" "$f" >/dev/null 2>&1 || true
  done
  for f in "$base/llm_seeds"/*."$seeds_ext"; do
    [ -f "$f" ] && LLVM_PROFILE_FILE="$tmp/llm_$(basename "$f").profraw" \
      "$HARNESS_COV" "$f" >/dev/null 2>&1 || true
  done

  # 3) Merge & report
  local prof="$trial_dir/coverage/coverage_$tsf.profdata"
  llvm-profdata merge -sparse "$tmp"/*.profraw -o "$prof" || true
  local rpt="$trial_dir/coverage/report_$tsf.txt"
  llvm-cov report "$HARNESS_COV" -instr-profile="$prof" >"$rpt" || true

  # 4) Extract TOTAL line percentages
  local TOTAL
  TOTAL=$(grep TOTAL "$rpt" || echo "")
  local LINES=$(echo "$TOTAL" | awk '{print $10}' | tr -d '%' || echo 0)
  local BRANCHES=$(echo "$TOTAL" | awk '{print $13}' | tr -d '%' || echo 0)
  local FUNCS=$(echo "$TOTAL" | awk '{print $7}'  | tr -d '%' || echo 0)

  # 5) Append to CSV
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date +'%Y-%m-%d %H:%M:%S')" "$exp" "$LINES" "$BRANCHES" "$FUNCS" \
    "$llm_count" "$corpus" \
    >>"$trial_dir/coverage/coverage_data.csv"

  rm -rf "$tmp"
}

start_coverage_tracking() {
  local exp=$1 base=$2 trial=$3 seeds_ext=$4 pid=$5
  while kill -0 "$pid" 2>/dev/null; do
    track_coverage "$exp" "$base" "$trial" "$seeds_ext"
    sleep "$TRACK_INTERVAL"
  done
}

# ———————— 4. Runners ————————

run_xml() {
  export HARNESS_COV="$XML_HARNESS_COV"
  for trial in $(seq 1 "$NUM_TRIALS"); do
    setup_trial_dirs "$XML_BASE" "$trial"
    export AFL_RANDOM_SEED=$((RANDOM_SEED_BASE + trial))
    afl-fuzz \
      -i "$XML_SEEDS_DIR" \
      -o "$XML_BASE/trial_$trial/output" \
      -V "$AFL_TIMEOUT" \
      -- "$XML_HARNESS" --noout @@ \
      >"$XML_BASE/trial_$trial/afl_output.log" 2>&1 &
    pid=$!
    start_coverage_tracking xml "$XML_BASE" "$trial" xml "$pid" &
    wait "$pid" || true
  done
}

run_sql() {
  export HARNESS_COV="$SQL_HARNESS_COV"
  for trial in $(seq 1 "$NUM_TRIALS"); do
    setup_trial_dirs "$SQL_BASE" "$trial"
    export AFL_RANDOM_SEED=$((RANDOM_SEED_BASE + trial))
    afl-fuzz \
      -i "$SQL_SEEDS_DIR" \
      -o "$SQL_BASE/trial_$trial/output" \
      -V "$AFL_TIMEOUT" \
      -- "$SQL_HARNESS" @@ \
      >"$SQL_BASE/trial_$trial/afl_output.log" 2>&1 &
    pid=$!
    start_coverage_tracking sql "$SQL_BASE" "$trial" test "$pid" &
    wait "$pid" || true
  done
}

# ———————— 5. Dispatch ————————
case "${1:-all}" in
  xml) run_xml    ;;
  sql) run_sql    ;;
  all) run_xml; run_sql ;;
  *)
    echo "Usage: $0 [xml|sql|all]" >&2
    exit 1
    ;;
esac
