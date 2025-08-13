import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def kl_divergence(p, q):
    """Compute the KL divergence between two distributions."""
    # Make sure that the distributions have the same keys
    # if not, add missing keys with zero probability
    for key in p.keys():
        if key not in q:
            raise ValueError(f"Key {key} not found in q")
            # q[key] = 0.0
    for key in q.keys():
        if key not in p:
            raise ValueError(f"Key {key} not found in p")
            # p[key] = 0.0
    # Compute the KL divergence
    kl = 0.0
    for key in p.keys():
        if p[key] > 0 and q[key] > 0:
            kl += p[key] * np.log(p[key] / q[key])
    return kl

def load_runs_data(run_dir: str, min_steps: int) -> list[dict]:
    runs = []
    # list subdirectories in the run_dir
    for run in os.listdir(run_dir):
        partial_run_path = os.path.join(run_dir, run)
        # check if it is a directory
        if os.path.isdir(partial_run_path):
            print(partial_run_path)
            # list all json files in the subdirectory
            for json_file in os.listdir(partial_run_path):
                # check if it is a json file
                if json_file.endswith(".json"):
                    # load the json file
                    with open(os.path.join(partial_run_path, json_file), "r") as f:
                        run_data = json.load(f)
                        # print(len(run_data["steps"]))
                        # runs.append(run_data)
                        if len(run_data["steps"]) >= min_steps:
                            runs.append(run_data)
    # print(len(runs))
    return runs

def load_mcmc_run_data(run_dir: str, min_steps: int) -> list[dict]:
    runs = []
    for run in os.listdir(run_dir):
        if run.endswith(".json"):
            with open(os.path.join(run_dir, run), "r") as f:
                run_data = json.load(f)
                if len(run_data["steps"]) >= min_steps:
                    runs.append(run_data)
    print(f"Loaded {len(runs)} samples from {run_dir}")
    return runs

def estimate_full_distribution(mcmc_samples: list[dict], distr_type: str) -> dict:
    if distr_type not in ["raw_logprob", "cons_logprob"]:
        raise ValueError("Invalid distribution type. Choose 'raw_logprob' or 'cons_logprob'.")
    # Initialize an empty dictionary to hold the distribution data
    logprobs = {}
    mismatches = 0
    mismatch_list = []
    max_diff = 0
    max_diff_rel = 0
    # Iterate over each MCMC run
    for mcmc_sample in mcmc_samples:
        # Extract the steps from the MCMC run
        steps = mcmc_sample["steps"]
        # Iterate over each step
        for step in steps:
            # Check both the current and proposed steps
            for side in ["current", "proposal"]:
            # for side in ["proposal"]:
                sample = step[side]
                # sample_str = "".join(sample["tokens"])
                sample_ids = tuple(sample["token_ids"])
                sample_tokens = tuple(sample["tokens"])
                sample_logprob = sample[distr_type]
                # Check if the sample is already in the dictionary
                if sample_ids not in logprobs:
                    # If not, initialize it with an empty list
                    logprobs[sample_ids] = sample_logprob
                else:
                    # assert logprobs[sample_str] == sample_logprob, f"Logprob mismatch for sample {sample_str}: {logprobs[sample_str]} vs {sample_logprob}"
                    # Check that the logprob is the same
                    if logprobs[sample_ids] != sample_logprob:
                        mismatches += 1
                        mismatch_list.append((sample_tokens, logprobs[sample_ids], sample_logprob))
                        max_diff = max(max_diff, abs(logprobs[sample_ids] - sample_logprob))
                        max_diff_rel = max(max_diff_rel, abs((logprobs[sample_ids] - sample_logprob) / logprobs[sample_ids]))
                        # raise ValueError(f"Logprob mismatch for sample {sample_str}: {distr_data[sample_str]} vs {sample_logprob}")
    print(  f"Number of mismatches: {mismatches}")
    print(  f"Max difference: {max_diff}")
    print(  f"Max relative difference: {max_diff_rel}")
    # sort the mismatch list by the absolute difference
    mismatch_list.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
    # print the top 10 mismatches
    for i in range(min(10, len(mismatch_list))):
        print(f"Mismatch {i}: {mismatch_list[i][0]}: {mismatch_list[i][1]} vs {mismatch_list[i][2]}")

    # # print sequences with highest logprobs
    # sorted_distr_data = sorted(logprobs.items(), key=lambda x: x[1], reverse=True)
    # # print top 10 sequences
    # for i in range(10):
    #     print(sorted_distr_data[i])

    total_logprob = -np.inf
    # Iterate over each sample in the distribution
    for logprob in logprobs.values():
        # Sum the logprobs
        total_logprob = np.logaddexp(total_logprob, logprob)
    # Normalize the logprobs
    normalized_logprobs = {sample_str: logprob - total_logprob for sample_str, logprob in logprobs.items()}
    # Convert the logprobs to probabilities
    distribution = {sample_str: float(np.exp(logprob)) for sample_str, logprob in normalized_logprobs.items()}
    print(sum(distribution.values()))
    assert np.isclose(sum(distribution.values()), 1.0), "Probabilities do not sum to 1"

    # # Print sequences with highest probabilities
    # sorted_probabilities = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    # # Print top 10 sequences
    # for i in range(10):
    #     print(sorted_probabilities[i])
    return distribution

# def get_mcmc_empirical_distribution(mcmc_runs: list[dict], n_steps: int, target_distr: dict) -> dict:
def get_mcmc_empirical_distribution(mcmc_samples: list[dict], n_steps: int) -> dict:
    counts = {}
    # Iterate over each MCMC run
    for mcmc_sample in mcmc_samples:
        # sample_at_step = "".join(mcmc_run["steps"][n_steps-1]["current"]["tokens"])
        sample_at_step = tuple(mcmc_sample["steps"][n_steps-1]["current"]["token_ids"])
        # Increment the count for the sample
        if sample_at_step not in counts:
            counts[sample_at_step] = 0
        counts[sample_at_step] += 1

    # # Print samples with highest counts
    # sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    # # Print top 10 sequences
    # for i in range(10):
    #     print(sorted_counts[i])

    # Normalize the counts to get the empirical distribution
    total_count = sum(counts.values())
    # Normalize the counts
    empirical_distribution = {sample: float(count / total_count) for sample, count in counts.items()}
    # Convert the counts to probabilities
    assert np.isclose(sum(empirical_distribution.values()), 1.0), "Probabilities do not sum to 1"
    # Print sequences with highest probabilities
    # sorted_probabilities = sorted(empirical_distribution.items(), key=lambda x: x[1], reverse=True)
    # # Print top 10 sequences
    # for i in range(10):
    #     print(sorted_probabilities[i])
    return empirical_distribution

def match_supports(mcmc_distr: dict, target_distr: dict, keep_support: str) -> tuple[dict, dict]:
    assert keep_support in ["mcmc", "target"], "keep_support must be either 'mcmc' or 'target'"
    # check that support of mcmc_distr is a subset of target_distr
    for sample in mcmc_distr:
        if sample not in target_distr:
            raise ValueError(f"Sample {sample} found in MCMC distribution but not in target distribution")

    res_mcmc_distr, res_target_distr = mcmc_distr.copy(), target_distr.copy()

    if keep_support == "mcmc":
        # we restrict the target distribution to the support of the MCMC distribution
        # and then renormalize it
        res_target_distr = {sample: target_distr[sample] for sample in mcmc_distr}
        total_count = sum(res_target_distr.values())
        res_target_distr = {sample: count / total_count for sample, count in res_target_distr.items()}
    elif keep_support == "target":
        # we add missing samples from the target distribution to the MCMC distribution
        # with zero probability
        for sample in target_distr:
            if sample not in res_mcmc_distr:
                res_mcmc_distr[sample] = 0.0
    
    # assert that both distributions sum to 1
    assert np.isclose(sum(res_mcmc_distr.values()), 1.0), "MCMC distribution does not sum to 1"
    assert np.isclose(sum(res_target_distr.values()), 1.0), "Target distribution does not sum to 1"

    return res_mcmc_distr, res_target_distr

def bootstrap_kl(samples, target_distr, n_bootstrap=10):
    """Perform bootstrap resampling to get confidence intervals for KL divergence."""
    n_samples = len(samples)
    bootstrap_kls = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        resampled = [samples[i] for i in resampled_indices]
        
        # Get empirical distribution from resampled data
        counts = {}
        for sample in resampled:
            if sample not in counts:
                counts[sample] = 0
            counts[sample] += 1
        
        # Normalize to get probabilities
        total = sum(counts.values())
        empirical_distr = {s: count/total for s, count in counts.items()}
        
        # Match supports and compute KL
        matched_mcmc, matched_target = match_supports(
            empirical_distr, target_distr, keep_support="target")
            # empirical_distr, target_distr, keep_support="mcmc")
        kl = kl_divergence(matched_mcmc, matched_target)
        bootstrap_kls.append(kl)
        
    # Calculate confidence intervals
    lower_ci = np.percentile(bootstrap_kls, 2.5)
    upper_ci = np.percentile(bootstrap_kls, 97.5)
    mean_kl = np.mean(bootstrap_kls)
    
    return mean_kl, lower_ci, upper_ci

def format_asap_run(asap_run_path: str, mcmc_run_dirs: list[str]) -> list[dict]:
    min_steps = 10

    with open(asap_run_path, "r") as f:
        asap_runs = f.read().splitlines()
        asap_runs = [json.loads(run) for run in asap_runs]
        # print(f"Loaded {len(asap_runs)} ASAP runs")
        # print(asap_runs[0][0]["tokens"][0])
        print(asap_runs[0][0]["raw_probability"])

    mcmc_runs = [load_mcmc_run_data(run_dir, min_steps=min_steps) for run_dir in mcmc_run_dirs]
    all_samples = [sample for run in mcmc_runs for sample in run]

    logprobs = {}
    for sample in all_samples:
        for step in sample["steps"]:
            # Check both the current and proposed steps
            for side in ["current", "proposal"]:
                sample = step[side]
                sample_ids = tuple(sample["token_ids"])
                sample_tokens = tuple(sample["tokens"])
                sample_logprob = sample["raw_logprob"]
                # Check if the sample is already in the dictionary
                if sample_ids not in logprobs:
                    # If not, initialize it with an empty list
                    logprobs[sample_ids] = sample_logprob
                else:
                    assert logprobs[sample_ids] == sample_logprob, f"Logprob mismatch for sample {sample_tokens}: {logprobs[sample_ids]} vs {sample_logprob}"
    
    # true_distr_est = estimate_full_distribution(all_samples, "raw_logprob")
    # print(len(true_distr_est))


    logprob_diffs = []
    formatted_asap_runs = []
    for asap_run in asap_runs:
        fmt_asp_run = {
            "steps": [],
        }
        for step in asap_run:
            step_token_ids = step["tokens"][0]
            step_raw_logprob = step["raw_probability"]

            # check if the step is in the logprobs
            if tuple(step_token_ids) not in logprobs:
                # print(f"Step {step_token_ids} not found in logprobs")
                logprob = step_raw_logprob
            else:
                logprob = logprobs[tuple(step_token_ids)]
                logprob_diffs.append(abs(step_raw_logprob - logprob))

            # mcmc_logprob = logprobs[tuple(step_token_ids)]
            # print(abs(step_raw_logprob - mcmc_logprob))
            # logprob_diffs.append(abs(step_raw_logprob - mcmc_logprob))

            fmt_step = {
                "current": {
                    "tokens": [],
                    "token_ids": step["tokens"][0],
                    "raw_logprob": logprob,
                    "cons_logprob": -1,
                },
                "proposal": {
                    "tokens": [],
                    "token_ids": step["tokens"][0],
                    "raw_logprob": logprob,
                    "cons_logprob": -1,
                },
            }
            fmt_asp_run["steps"].append(fmt_step)
        #print stats about the logprob diffs
        formatted_asap_runs.append(fmt_asp_run)
    print(f"Mean logprob diff: {np.mean(logprob_diffs)}")
    print(f"Max logprob diff: {np.max(logprob_diffs)}")
    print(f"Min logprob diff: {np.min(logprob_diffs)}")
    print(f"Std logprob diff: {np.std(logprob_diffs)}")

    return formatted_asap_runs



def plot_kl_runs(mcmc_run_dirs: list[str], task_id: str, output_dir: str):
    steps_total = 10
    mcmc_runs = [load_mcmc_run_data(run_dir, min_steps=steps_total) for run_dir in mcmc_run_dirs]

    # flatten mcmc_runs
    all_samples = [sample for run in mcmc_runs for sample in run]
    true_distr_est = estimate_full_distribution(all_samples, "raw_logprob")
    print(len(true_distr_est))

    runs_kls = []
    steps_range = list(range(1, steps_total+1))
    
    for run in mcmc_runs:
        run_kls = []
        run_lower_cis = []
        run_upper_cis = []
        n_samples_in_run = len(run)
        
        for n_steps in steps_range:
            # Get samples at this step
            samples = [tuple(step["steps"][n_steps-1]["current"]["token_ids"]) 
                       for step in run]
            
            # Compute KL divergence with bootstrapping
            # mean_kl, lower_ci, upper_ci = bootstrap_kl(samples, true_distr_est, n_bootstrap=n_samples_in_run)
            mean_kl, lower_ci, upper_ci = bootstrap_kl(samples, true_distr_est, n_bootstrap=500)
            
            run_kls.append(mean_kl)
            run_lower_cis.append(lower_ci)
            run_upper_cis.append(upper_ci)
        
        runs_kls.append((run_kls, run_lower_cis, run_upper_cis))

    # Create a figure and axis
    plt.figure(figsize=(6, 6))

    # Get a colormap with distinct colors
    cmap = get_cmap('tab10')

    # Plot each run with a different color
    for i, ((run_kls, lower_cis, upper_cis), run_dir) in enumerate(zip(runs_kls, mcmc_run_dirs)):
        # Extract just the directory name for cleaner labels
        label = os.path.basename(run_dir).rsplit("-", 1)[-1]
        if label == "prefix":
            label = "uniform"
        color = cmap(i)
        
        # Plot the mean KL divergence
        plt.plot(steps_range, run_kls, marker='o', linestyle='-', linewidth=2, 
                 color=color, label=f"{label}")
        
        # Plot the confidence intervals
        plt.fill_between(steps_range, lower_cis, upper_cis, 
                         alpha=0.2, color=color, label='_nolegend_')

    # Add decorations
    plt.xlabel('k', fontsize=18)
    plt.ylabel('KL Divergence', fontsize=18)
    # plt.title('KL Divergence vs. Steps', fontsize=16)
    plt.title(task_id, fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='best', framealpha=0.7)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{task_id}-kl_div.png"), dpi=200)

def print_kl_stats(tasks_at_0, tasks_at_10):
    methods = ["prefix", "priority", "restart"]
    for method in methods:
        improvement_ratios = []
        for kl_0, kl_10 in zip(tasks_at_0[method], tasks_at_10[method]):
            # impr = (kl_0 - kl_10) / kl_0
            impr = kl_0 / kl_10
            # if kl_0 == 0:
            #     impr = .01
            # else:
            #     impr = kl_10 / kl_0
                
            improvement_ratios.append(impr)
        # take geometric mean
        improvement_ratio = np.prod(improvement_ratios) ** (1 / len(improvement_ratios))
        print(f"Improvement ratio for {method}: {improvement_ratio:.4f}")

def print_asap_kl_stats(tasks_at_10):
    methods = ["prefix", "priority", "restart"]
    for method in methods:
        improvement_ratios = []
        for kl_0, kl_10 in zip(tasks_at_10["asap"], tasks_at_10[method]):
            # print(kl_0, kl_10)
            # impr = (kl_0 - kl_10) / kl_0
            impr = kl_0 / kl_10
            # if kl_0 == 0:
            #     impr = .01
            # else:
            #     impr = kl_10 / kl_0
                
            improvement_ratios.append(impr)
        # take geometric mean
        # print(improvement_ratios)
        improvement_ratio = np.prod(improvement_ratios) ** (1 / len(improvement_ratios))
        print(f"Improvement ratio for {method}: {improvement_ratio:.4f}")


def plot_kl_scatter(all_mcmc_run_data: list[tuple[list[str], str]], split: str, output_dir: str):
    # for each run, plot KL divergence as a scatter
    # x axis is kl at step 0, y axis is kl at step 10
    # each run is a different color
    steps_total = 10

    task_kls_at_0 = {
        "prefix": [],
        "priority": [],
        "restart": []
    }
    task_kls_at_10 = {
        "prefix": [],
        "priority": [],
        "restart": []
    }

    for mcmc_run_dirs, task_id in all_mcmc_run_data:

        mcmc_runs = [load_mcmc_run_data(run_dir, min_steps=steps_total) for run_dir in mcmc_run_dirs]

        # flatten mcmc_runs
        all_samples = [sample for run in mcmc_runs for sample in run]
        true_distr_est = estimate_full_distribution(all_samples, "raw_logprob")
        print(len(true_distr_est))
        # run_kls = {}
        for r, run in enumerate(mcmc_runs):
            # Extract just the directory name for cleaner labels
            label = os.path.basename(mcmc_run_dirs[r]).rsplit("-", 1)[-1]
            if label == "asap":
                continue

            samples_step_0 = [tuple(step["steps"][0]["current"]["token_ids"]) for step in run]
            samples_step_10 = [tuple(step["steps"][steps_total - 1]["current"]["token_ids"]) for step in run]
            mean_kl_step_0, _, _ = bootstrap_kl(samples_step_0, true_distr_est, n_bootstrap=500)
            mean_kl_step_10, _, _ = bootstrap_kl(samples_step_10, true_distr_est, n_bootstrap=500)
            # run_kls[label] = (mean_kl_step_0, mean_kl_step_10)
            task_kls_at_0[label].append(mean_kl_step_0)
            task_kls_at_10[label].append(mean_kl_step_10)

            # Convert to numpy arrays for scatter plot
            kl_step_0 = np.array([mean_kl_step_0])
            kl_step_10 = np.array([mean_kl_step_10])
            # Remove NaN values
            kl_step_0 = kl_step_0[~np.isnan(kl_step_0)]
            kl_step_10 = kl_step_10[~np.isnan(kl_step_10)]
            # Scatter plot: KL divergence at step 1 vs KL divergence at step 10
    

    # Create a figure with equal aspect ratio (square)
    plt.figure(figsize=(6, 6))
    # Get a colormap with distinct colors
    cmap = get_cmap('tab10')

    # Find min and max values for both axes to set equal scale
    min_val = min(min(task_kls_at_0["restart"]), min(task_kls_at_10["prefix"]), min(task_kls_at_10["priority"]), min(task_kls_at_10["restart"]))
    max_val = max(max(task_kls_at_0["restart"]), max(task_kls_at_10["prefix"]), max(task_kls_at_10["priority"]), max(task_kls_at_10["restart"]))

    # Add some padding
    padding = (max_val - min_val) * 0.05
    plot_range = [min_val - padding, max_val + padding]

    # Plot the scatter points
    plt.scatter(task_kls_at_0["restart"], task_kls_at_10["prefix"], s=100, color=cmap(0), edgecolors="none", label="uniform", alpha=0.7)
    plt.scatter(task_kls_at_0["restart"], task_kls_at_10["priority"], s=100, color=cmap(1), edgecolors="none", label="priority", alpha=0.7)
    plt.scatter(task_kls_at_0["restart"], task_kls_at_10["restart"], s=100, color=cmap(2), edgecolors="none", label="restart", alpha=0.7)

    # Add diagonal line (y=x)
    plt.plot(plot_range, plot_range, 'k:', alpha=0.7)
    
    # Set the same limits for both axes
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    
    # Add decorations
    plt.xlabel('GCD KL Divergence', fontsize=18)
    plt.ylabel(f'MCMC(k=10) KL Divergence', fontsize=18)
    plt.title(f'{split}', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='best', framealpha=0.7)
    plt.tight_layout()
    plt.gca().set_aspect('equal')  # Ensure perfect square aspect ratio

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{split}-kl-gcd_scatter.png"), dpi=200)

    print_kl_stats(task_kls_at_0, task_kls_at_10)

def plot_kl_scatter_asap(all_mcmc_run_data: list[tuple[list[str], str]], split: str, output_dir: str):
    # for each run, plot KL divergence as a scatter
    # x axis is kl at step 0, y axis is kl at step 10
    # each run is a different color
    steps_total = 10

    task_kls_at_0 = {
        "prefix": [],
        "priority": [],
        "restart": [],
        "asap": [],
    }
    task_kls_at_10 = {
        "prefix": [],
        "priority": [],
        "restart": [],
        "asap": [],
    }

    for mcmc_run_dirs, task_id in all_mcmc_run_data:

        mcmc_runs = [load_mcmc_run_data(run_dir, min_steps=steps_total) for run_dir in mcmc_run_dirs]

        # flatten mcmc_runs
        all_samples = [sample for run in mcmc_runs for sample in run]
        true_distr_est = estimate_full_distribution(all_samples, "raw_logprob")
        print(len(true_distr_est))
        # run_kls = {}
        for r, run in enumerate(mcmc_runs):
            # Extract just the directory name for cleaner labels
            label = os.path.basename(mcmc_run_dirs[r]).rsplit("-", 1)[-1]

            samples_step_0 = [tuple(step["steps"][0]["current"]["token_ids"]) for step in run]
            samples_step_10 = [tuple(step["steps"][steps_total - 1]["current"]["token_ids"]) for step in run]
            mean_kl_step_0, _, _ = bootstrap_kl(samples_step_0, true_distr_est, n_bootstrap=500)
            mean_kl_step_10, _, _ = bootstrap_kl(samples_step_10, true_distr_est, n_bootstrap=500)
            # run_kls[label] = (mean_kl_step_0, mean_kl_step_10)
            task_kls_at_10[label].append(mean_kl_step_10)
            task_kls_at_0[label].append(mean_kl_step_0)

            # Convert to numpy arrays for scatter plot
            kl_step_0 = np.array([mean_kl_step_0])
            kl_step_10 = np.array([mean_kl_step_10])
            # Remove NaN values
            kl_step_0 = kl_step_0[~np.isnan(kl_step_0)]
            kl_step_10 = kl_step_10[~np.isnan(kl_step_10)]
            # Scatter plot: KL divergence at step 1 vs KL divergence at step 10
    

    # Create a figure with equal aspect ratio (square)
    plt.figure(figsize=(6, 6))
    # Get a colormap with distinct colors
    cmap = get_cmap('tab10')

    # Find min and max values for both axes to set equal scale
    # min_val = min(min(task_kls_at_0["prefix"]), min(task_kls_at_10["prefix"]))
    min_val = min(min(task_kls_at_10["prefix"]), min(task_kls_at_10["priority"]), min(task_kls_at_10["restart"]), min(task_kls_at_10["asap"]))
    # max_val = max(max(task_kls_at_0["prefix"]), max(task_kls_at_10["prefix"]))
    max_val = max(max(task_kls_at_10["prefix"]), max(task_kls_at_10["priority"]), max(task_kls_at_10["restart"]), max(task_kls_at_10["asap"]))

    # Add some padding
    padding = (max_val - min_val) * 0.05
    plot_range = [min_val - padding, max_val + padding]

    # Plot the scatter points
    plt.scatter(task_kls_at_10["asap"], task_kls_at_10["prefix"], s=100, color=cmap(0), edgecolors='none', label="uniform", alpha=0.7)
    plt.scatter(task_kls_at_10["asap"], task_kls_at_10["priority"], s=100, color=cmap(1), edgecolors='none', label="priority", alpha=0.7)
    plt.scatter(task_kls_at_10["asap"], task_kls_at_10["restart"], s=100, color=cmap(2), edgecolors='none', label="restart", alpha=0.7)

    # Add diagonal line (y=x)
    plt.plot(plot_range, plot_range, 'k:', alpha=0.7)
    
    # Set the same limits for both axes
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    
    # Add decorations
    plt.xlabel('ASAP(k=10) KL Divergence', fontsize=18)
    plt.ylabel(f'MCMC(k=10) KL Divergence', fontsize=18)
    # plt.title(f'ASAP@10 vs MCMC@10 KL Divergence, {split}', fontsize=20)
    plt.title(f'{split}', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='best', framealpha=0.7)
    plt.tight_layout()
    plt.gca().set_aspect('equal')  # Ensure perfect square aspect ratio

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{split}-kl-asap_scatter.png"), dpi=200)

    # print_kl_stats(task_kls_at_0, task_kls_at_10)
    print_asap_kl_stats(task_kls_at_10)
    

def plot_sampled_mass(mcmc_run_dirs: list[str]):
    steps_total = 20
    mcmc_runs = [load_mcmc_run_data(run_dir, min_steps=steps_total) for run_dir in mcmc_run_dirs]

    runs_mass_stats = []
    steps_range = list(range(1, steps_total+1))
    for run in mcmc_runs:
        # print(f"Processing run with {len(run)} samples")
        # sampled_mass = []
        avg_logprobs = []
        avg_logprobs_wor = []
        for n_steps in steps_range:
            logprobs = [
                (tuple(step["steps"][n_steps-1]["current"]["token_ids"]), 
                 step["steps"][n_steps-1]["current"]["raw_logprob"])
                for step in run
            ]
            avg_logprob = np.mean([logprob for _, logprob in logprobs])

            samples_wor = {}
            for seq, logprob in logprobs:
                if seq not in samples_wor:
                    samples_wor[seq] = logprob
                else:
                    assert samples_wor[seq] == logprob, f"Logprob mismatch for sample {seq}: {samples_wor[seq]} vs {logprob}"
            
            avg_logprob_wor = np.mean(list(samples_wor.values()))

            avg_logprobs.append(avg_logprob)
            avg_logprobs_wor.append(avg_logprob_wor)
        runs_mass_stats.append((avg_logprobs, avg_logprobs_wor))

    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Get a colormap with distinct colors
    cmap = get_cmap('tab10')

    # Plot each run with a different color
    for i, ((avg_logprobs, avg_logprobs_wor), run_dir) in enumerate(zip(runs_mass_stats, mcmc_run_dirs)):
        # Extract just the directory name for cleaner labels
        label = os.path.basename(run_dir)
        
        # Plot average logprobs with solid line
        plt.plot(steps_range, avg_logprobs, marker='o', linestyle='-', linewidth=2, 
                 color=cmap(i), label=f"{label} (with rep.)")
        
        # Plot average logprobs without repetition with dashed line
        plt.plot(steps_range, avg_logprobs_wor, marker='o', linestyle='--', linewidth=2, 
                 color=cmap(i), label=f"{label} (w/o rep.)")

    # Add decorations
    plt.xlabel('Number of MCMC Steps', fontsize=14)
    plt.ylabel('Average Log Probability', fontsize=14)
    plt.title('Average Log Probability vs. Steps', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best', framealpha=0.7)
    plt.tight_layout()

    plt.show()

def get_mcmc_samples(mcmc_runs: list[dict], n_steps: int) -> list[tuple]:
    samples = []
    for mcmc_run in mcmc_runs:
        sample_at_step = tuple(mcmc_run["steps"][n_steps-1]["current"]["token_ids"])
        logprob_at_step = mcmc_run["steps"][n_steps-1]["current"]["raw_logprob"]
        samples.append((sample_at_step, logprob_at_step))
    return samples