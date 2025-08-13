import os
import json
import time
import gc
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

import lib
import utils

def is_valid_propose_style(propose_style):
    if propose_style in ["prefix", "priority", "restart"]:
        return True
    mix, p = propose_style.split("-")
    p = float(p)
    if mix == "mix" and 0 <= p <= 1:
        return True
    return False

class MCMC:
    def __init__(
        self, 
        model: lib.ConstrainedModel, 
        prompt: str, 
        propose_style: str,
        name_prefix: str,
        root_log_dir: str, 
    ):
        self.model = model
        prompt = model._format_prompt(prompt)
        self.prompt_ids = model.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.model.device)
        # assert propose_style in ["prefix", "priority", "restart"]
        assert is_valid_propose_style(propose_style)
        self.propose_style = propose_style
        self.root_log_dir = root_log_dir
        os.makedirs(root_log_dir, exist_ok=True)
        self.log_dir = f"{root_log_dir}/{utils.timestamp()}-{name_prefix}-{propose_style}"
        os.makedirs(self.log_dir, exist_ok=True)

    def get_sample(self, n_steps: int, max_new_tokens: int):
        # hopefully this works
        gc.collect()
        torch.cuda.empty_cache()

        current_ids, current_scores = self.model._generate(
            self.prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            constrain=True,
            prefix_ids=None,
        )
        current_cons_logprob = self.model._get_seq_logprob_from_scores(current_scores, current_ids).item()
        current_raw_logprob = self.model._get_seq_logprob(self.prompt_ids, current_ids, constrain=False).item()
        print(f"Initial: {[self.model.tokenizer.decode(token_id) for token_id in current_ids[0]]}")

        steps = []
        sample_file = f"{self.log_dir}/{utils.timestamp(millis=True)}-n{n_steps}.json"

        for i in range(n_steps):
            step_propose_style = self.propose_style
            if step_propose_style.startswith("mix"):
                _, p = step_propose_style.split("-")
                p = float(p)
                step_propose_style = "restart" if np.random.rand() < p else "priority"
            print(f"Step {i} ({step_propose_style})")

            print(f"Current: {[self.model.tokenizer.decode(token_id) for token_id in current_ids[0]]}")
            print(f"Current raw logprob: {current_raw_logprob}")
            proposal_ids, proposal_scores, _ = self.model._propose_next_sequence(
                prompt_ids=self.prompt_ids,
                current_ids=current_ids,
                max_new_tokens=max_new_tokens,
                constrain=True,
                current_scores=current_scores,
                propose_style=step_propose_style,
            )
            proposal_raw_logprob = self.model._get_seq_logprob(self.prompt_ids, proposal_ids, constrain=False).item()
            proposal_cons_logprob = self.model._get_seq_logprob_from_scores(proposal_scores, proposal_ids).item()
            print(f"Proposal: {[self.model.tokenizer.decode(token_id) for token_id in proposal_ids[0]]}")
            print(f"Proposal raw logprob: {proposal_raw_logprob}")

            acceptance_prob = None
            if torch.equal(current_ids, proposal_ids):
                acceptance_prob = 1
            else:
                prop_logprob_cur_to_next = self.model._propose_next_sequence_logprob(
                    current_ids=current_ids,
                    current_scores=current_scores,
                    next_ids=proposal_ids,
                    next_scores=proposal_scores,
                    propose_style=step_propose_style,
                )

                prop_logprob_next_to_cur = self.model._propose_next_sequence_logprob(
                    current_ids=proposal_ids,
                    current_scores=proposal_scores,
                    next_ids=current_ids,
                    next_scores=current_scores,
                    propose_style=step_propose_style,
                )

                log_acc_ratio = proposal_raw_logprob + prop_logprob_next_to_cur - \
                    current_raw_logprob - prop_logprob_cur_to_next

                acceptance_prob = min(1, np.exp(log_acc_ratio))
            print(f"Acceptance prob: {acceptance_prob}")
    
            accepted = bool(np.random.rand() < acceptance_prob)

            # save to steps
            step = {
                "current": {
                    "tokens": [self.model.tokenizer.decode(token_id) for token_id in current_ids[0]],
                    "token_ids": [int(id) for id in current_ids[0]],
                    "raw_logprob": current_raw_logprob,
                    "cons_logprob": current_cons_logprob,
                },
                "proposal": {
                    "tokens": [self.model.tokenizer.decode(token_id) for token_id in proposal_ids[0]],
                    "token_ids": [int(id) for id in proposal_ids[0]],
                    "raw_logprob": proposal_raw_logprob,
                    "cons_logprob": proposal_cons_logprob,
                },
                "acceptance_prob": acceptance_prob,
                "accepted": accepted,
            }
            steps.append(step)
            steps_dump = {"steps": steps}
            with open(sample_file, "w") as f:
                json.dump(steps_dump, f, indent=4)

            if accepted:
            # if np.random.rand() < acceptance_prob:
                current_ids = proposal_ids
                current_scores = proposal_scores
                current_cons_logprob = proposal_cons_logprob
                current_raw_logprob = proposal_raw_logprob
                print("Accepted")
            
            print("\n\n")
            
        return current_ids

    def get_samples(self, n_samples: int, n_steps: int, max_new_tokens: int):
        for i in tqdm(range(n_samples)):
            print(f"\nSample {i}")
            sample_start_time = time.time()
            sample = self.get_sample(n_steps, max_new_tokens)
            sample_end_time = time.time()
            sample_time = sample_end_time - sample_start_time
            print(f"Sample time: {sample_time:.2f} s")
            sample_str = self.model.tokenizer.decode(sample[0])
            print(f"Sample: {sample_str}")