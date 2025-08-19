import json
import os
from pathlib import Path
import glob

def write_tokens_to_file(filename: Path, tokens_to_write: list):
    """
    Writes a list of string tokens to a file after cleaning it.

    Args:
        filename (Path): The full path to the output file.
        tokens_to_write (list): The list of string tokens from the JSON.
    """
    if not tokens_to_write:
        print(f"WARNING: Token list is empty. Skipping file {filename}.")
        return

    content_tokens = tokens_to_write

    if content_tokens[-1] == '<|eot_id|>':
        content_tokens = content_tokens[:-1]

    file_content = "".join(content_tokens)
    
    try:
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(file_content)
        print(f"Successfully wrote {filename}")

    except IOError as e:
        print(f"ERROR: Could not write to file {filename}. Reason: {e}")


def process_json_file(json_path: Path, output_dirs, file_extension):
    """
    Loads a single JSON file, extracts proposals, and writes them to XML files.
    
    Args:
        json_path (Path): The path to the input JSON file.
    """
    print(f"\nProcessing file: {json_path.name}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {json_path}. Skipping.")
        return
    except FileNotFoundError:
        print(f"ERROR: File not found {json_path}. Skipping.")
        return

    try:
        proposals = [step["proposal"]["tokens"] for step in data["steps"]]
        
        file_stem = json_path.stem

        # Extract and write GCD
        if len(proposals) > 1:
            gcd_tokens = proposals[0]
            gcd_filename = output_dirs['gcd'] / f"{file_stem}_gcd.{file_extension}"
            write_tokens_to_file(gcd_filename, gcd_tokens)

        # Extract and write MCMC 2
        if len(proposals) > 2:
            mcmc_2_tokens = proposals[1]
            mcmc_2_filename = output_dirs['mcmc_2'] / f"{file_stem}_mcmc_2.{file_extension}"
            write_tokens_to_file(mcmc_2_filename, mcmc_2_tokens)
        else:
            print("WARNING: Not enough proposals for mcmc_2")

        # Extract and write MCMC 5
        if len(proposals) > 5:
            mcmc_5_tokens = proposals[4]
            mcmc_5_filename = output_dirs['mcmc_5'] / f"{file_stem}_mcmc_5.{file_extension}"
            write_tokens_to_file(mcmc_5_filename, mcmc_5_tokens)
        else:
            print("WARNING: Not enough proposals for mcmc_5")

        # Extract and write MCMC 10            
        if len(proposals) > 9:
            mcmc_10_tokens = proposals[9]
            mcmc_10_filename = output_dirs['mcmc_10'] / f"{file_stem}_9_mcmc_10.{file_extension}"
            write_tokens_to_file(mcmc_10_filename, mcmc_10_tokens)
        else:
            print("WARNING: Not enough proposals for mcmc_10")
            
    except (KeyError, TypeError) as e:
        print(f"ERROR: JSON structure is invalid in {json_path}. Missing key: {e}. Skipping.")

def extract_fuzz_files(mcmc_type, file_extension):
	script_dir = Path(__file__).resolve().parent
	base_search_path = script_dir / 'fuzz_runs'
 
	if file_extension == 'test':
		benchmark_type = 'sql'
	else:
		benchmark_type = file_extension

	output_base_dir = script_dir / 'benchmarks' / 'seeds' / file_extension
	output_dirs = {
		'gcd': Path(output_base_dir) / 'gcd' / 'llm_seeds',
		'mcmc_2': Path(output_base_dir) / 'mcmc_restart_2' / 'llm_seeds',
		'mcmc_5': Path(output_base_dir) / 'mcmc_restart_5' / 'llm_seeds',
		'mcmc_10': Path(output_base_dir) / 'mcmc_priority_10' / 'llm_seeds'
	}
 
	search_pattern = os.path.join(base_search_path, f'*/*{benchmark_type}-{mcmc_type}*', '*.json')
	print(f"\nSearching for JSON files in: {search_pattern}")
	json_files = [Path(p) for p in glob.glob(search_pattern)]
 
	if not json_files:
		print("\nNo JSON files found matching the criteria. Please check your paths and MCMC type.")
		return

	print(f"\nFound {len(json_files)} files to process.")

	# Process each file found
	for json_path in json_files:
		process_json_file(json_path, output_dirs, file_extension)
		
	print("\nScript finished.")

def main():
    """
    Main function to find and process all relevant JSON files.
    """
    config = {
		'priority': 'xml',
		'restart': 'xml',
		'prefix': 'xml',
	}
    
    for mcmc_type, file_extension in config.items():
        extract_fuzz_files(mcmc_type, file_extension)

    config = {
		'priority': 'test',
		'restart': 'test',
		'prefix': 'test',
	}
    
    for mcmc_type, file_extension in config.items():
        extract_fuzz_files(mcmc_type, file_extension)
        
if __name__ == "__main__":
    main()
