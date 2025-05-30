# Supplementary Fuzzing Material

This repository contains supplementary artifacts for the fuzzing runs. You will find,

- The **seed corpora** used to initialize each fuzzing variant
- The **experiment outputs** (coverage data, figures) for both SQL and XML benchmarks
- **Scripts** to reproduce the builds, runs, and coverage harnesses
- A Jupyter **notebook** for generating comparison plots

## System Requirements

- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **CPU**: Multi-core x86_64 processor (experiments used 16-core machines)
- **Memory**: 8GB+ RAM recommended
- **Storage**: ~5GB free space for full experiment data
- **Dependencies**: Python 3.8+, clang-11+, make, git

## Repository layout

```
├── benchmarks/
│ ├── seeds/
│ │ ├── sql/… ← initial `.test` files for each SQL variant
│ │ └── xml/… ← initial `.xml` files for each XML variant
│ ├── experiment_data/
| | ├── sql/
│ │ | ├─── <variant>/
│ │ | | ├─── trial1/
│ │ │ | | ├─── coverage_data.csv
│ │ | | | | … ← trial2, trial3, …
│ | └── xml/
│ | └── <variant>/… ← same structure as `sql/`
│ |
├── scripts/
| ├── experiments.conf ← contains the set of methods to run experiments for
│ ├── setup_dependencies.sh ← installs system deps & builds AFL++
│ ├── setup_fuzzing.sh ← builds libxml2 & SQLite with AFL++
│ ├── setup_coverage_harness.sh ← builds coverage-instrumented harnesses
│ ├── run_fuzzing_experiment.sh ← runs afl-fuzz + coverage tracking
│ └── main.sh ← top-level orchestrator (`[xml|sql|all]`)
├── grammars/
│ |── sqlite_test.ebnf ← grammar used for generating SQLite test cases
│ └── xml.ebnf ← grammar used for generating XML documents
|
├── prompts/
│ |── sql_gen.txt ← prompt used for SQL experiments
│ └── xml_gen.txt ← prompt used for XML experiments
|
├── notebooks/
│ └── visualize_comparison.ipynb ← interactive plotting
│
├── figures/
├── requirements.txt ← Python deps for notebook
└── README.md ← this file

```

## Experiment Configuration

### AFL++ Parameters

- **Version**: AFL++ 4.00c
- **Trial duration**: 3600 seconds (1 hour) per trial
- **Trials per method**: 5 independent runs
- **Seeds per corpus**: 100 seeds
- **Random seeds**: 42+i (i=1...5) for reproducibility

### Environment Variables

The following AFL++ environment variables are automatically configured:

- `AFL_RANDOM_SEED`: Trial-specific PRNG seeds (42-46)
- `AFL_SKIP_CPUFREQ=1`: Disable CPU frequency checks
- `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1`: Suppress crash warnings
- `AFL_NO_UI=1`: Non-interactive mode

### Coverage Instrumentation

Targets are built with LLVM instrumentation:

```bash
clang -fprofile-instr-generate -fcoverage-mapping target.c -o target
```

This adds ≤2% runtime overhead and enables precise branch coverage measurement.

## Getting started

1. **Install Python deps** (for the notebook) - (Optional)

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the full pipeline**
   _(Note: Set `$DATA_DIR` before starting your experiment!)_
   By default this will build everything, run all SQL and XML trials, and collect coverage:

   ```bash
   chmod +x scripts/*.sh
   ./scripts/main.sh all
   ```

   Or target just one domain:

   ```bash
   ./scripts/main.sh xml
   ./scripts/main.sh sql
   ```

   _Custom configuration_: Edit `scripts/experiments.conf` to modify which methods are evaluated.

3. **Visualize plots**

- _Prerequisites: If you don't have Jupyter installed, run `pip install -r requirements.txt`_
- Open `notebooks/coverage_comparison.ipynb` in Jupyter.

## Understanding the Output

### Coverage Data Format

Each trial produces `coverage_data.csv` with columns:

- `timestamp`: Seconds elapsed since fuzzing start
- `branches_covered`: Cumulative branch coverage count
- `total_branches`: Total branches in target binary

### Expected Runtime

- **Full reproduction**: ~120-121 hours hours (12 methods x 2 domains x 5 trials )
- **Single method**: ~5-6 hours (5 trials × 1 hour + setup time)
- **Quick test**: Use shorter durations by modifying `run_fuzzing_experiment.sh`

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure all scripts are executable (`chmod +x scripts/*.sh`)
2. **AFL++ build failures**: Check that clang-11+ is installed
3. **Out of disk space**: Each full run generates ~1-2GB of data
4. **CPU governor warnings**: Set `AFL_SKIP_CPUFREQ=1` (automatically handled)

## Directory conventions

- `benchmarks/seeds/…`
  Contains the _initial_ seed inputs for each experiment—no trial subfolders. Reviewers can reuse these to re-run any variant.

- `benchmarks/experiment_data/…`
  Contains the _outputs_ for each trial of each variant:

  - `coverage_data.csv` holds the time-series CSV

- `scripts/`
  All automation and build/run scripts.

  - `main.sh` ties everything together
  - `*.sh` are the step-by-step builders and runners
  - `experiments.conf` set of methods for which experiments are run

- `grammars/`
  The grammars used for generating the seeds.

- `prompts/`
  The prompts used in combination with the grammars for seed generation.

- `notebooks/`
  The original `.ipynb` for exploratory data analysis and plotting.

- `figures/`
  Static `.png` images that appear in the paper or supplement. These are regenerated by the script above.
