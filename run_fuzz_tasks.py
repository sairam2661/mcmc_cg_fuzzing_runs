import json
import os

import torch
from tqdm import tqdm

import utils
import lib
import mcmc

def load_fuzz_tasks():
	runs_params = [
		("xml", "prompts/xml_gen.txt", "grammars/xml.ebnf"),
		("sql", "prompts/sql_gen.txt", "grammars/sql.ebnf")
	]
 
	tasks = []
	
	for run_type, prompt_file, grammar_file in runs_params:
		with open(prompt_file, "r") as f:
			prompt = f.read().strip()
							
		grammar_str = open(grammar_file).read()
  
		tasks.append({
			"id": run_type,
			"prompt": prompt,
			"grammar": grammar_str,
		})
	return tasks
	
	
def run_mcmc_fuzz_tasks(benchmark):
	model_id = "meta-llama/Llama-3.1-8B-Instruct"

	model = lib.ConstrainedModel(model_id, None, torch_dtype=torch.bfloat16)

	root_log_dir = "fuzz_runs"

	split_log_dir = f"{root_log_dir}/{utils.timestamp()}-{benchmark}"
	# make sure the directory exists
	os.makedirs(split_log_dir, exist_ok=True)

	n_samples = 100
	n_steps = 10
	max_new_tokens = 512
	propose_styles = ["restart", "priority", "prefix"]
 
	tasks = load_fuzz_tasks()
	benchmark_tasks = [task for task in tasks if task["id"] == benchmark]

	for task in tqdm(benchmark_tasks):
		task_id = task["id"] 
		task_prompt = task["prompt"]
		task_grammar = task["grammar"]
		print(f"Benchmark: {task_id}")

		model._set_grammar_constraint(task_grammar)
		for propose_style in propose_styles:
			print(f"Benchmark: {task_id}, Propose Style: {propose_style}")

			model._set_grammar_constraint(task_grammar)
			mcmc_runner = mcmc.MCMC(
				model=model,
				prompt=task_prompt,
				propose_style=propose_style,
				name_prefix=task_id,
				root_log_dir=split_log_dir,
			)
			mcmc_runner.get_samples(
				n_samples=n_samples,
				n_steps=n_steps,
				max_new_tokens=max_new_tokens,
			)


if __name__ == "__main__":
	from argparse import ArgumentParser

	parser = ArgumentParser()
	parser.add_argument("--benchmark", required=True, choices=["xml", "sql"])

	args = parser.parse_args()
	benchmark = args.benchmark
	print(f"Benchmark: {benchmark}")

	run_mcmc_fuzz_tasks(benchmark)