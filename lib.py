import os
import json
import gc

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

def scores_to_top_k_tokens(scores, k):
    result = []
    for step_i, step_scores in enumerate(scores):
        probs = torch.log_softmax(step_scores, dim=-1)
        top_probs, top_token_ids = torch.topk(probs, k=k)
        top_probs = top_probs.tolist()
        # top_tokens = tokenizer.batch_decode(top_token_ids)
        top_choices = list(zip(top_token_ids, top_probs))
        result.append(top_choices)
    return result

class ConstrainedModel():
    HF_CHAT_MODELS = [
        # Llama
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        # Qwen
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        # DeepSeek
        # "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        # Microsoft
        # "microsoft/Phi-4-mini-instruct",
        "microsoft/Phi-3.5-mini-instruct",
        # Google
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        # Mistral
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.1",

    ]
    HF_BASE_MODELS = [
    ]

    def __init__(self, model_id: str, grammar_str: str | None = None, **kwargs):
        with open("secrets.json") as f:
            secrets = json.load(f)
            os.environ["HF_TOKEN"] = secrets["HF_TOKEN"]

        self.model_id = model_id

        # self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # self.config = AutoConfig.from_pretrained(model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Tokenizer: {self.tokenizer.name_or_path}")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto"
        # device_map = "balanced_low_0"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            # config=self.config, 
            device_map=device_map, 
            # trust_remote_code=True,
            **kwargs
        )
        # self.model = AutoModelForCausalLM.from_pretrained(model_id, config=self.config).to("cpu")
        print(f"Model: {self.model.name_or_path}")
        self.model.eval()

        print(f"Model device: {self.model.device}")
        if grammar_str is not None:
            self._set_grammar_constraint(grammar_str)

    def _set_grammar_constraint(self, grammar_str: str):
        self.grammar_constraint = IncrementalGrammarConstraint(grammar_str, "root", self.tokenizer)
        
    def _format_prompt(self, prompt: str) -> str:
        """
        Formats the prompt accordingly if it is a chat model.
        """
        if self.model_id in self.HF_BASE_MODELS:
            return prompt
        elif self.model_id in self.HF_CHAT_MODELS:
            messages = [
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            assert type(formatted_prompt) == str
            return formatted_prompt
        else:
            raise ValueError(f"Unknown model type for model {self.model_id}")

    def _unbatch_sequences(self, sequences: torch.Tensor) -> list[torch.Tensor]:
        """
        Unbatch the sequences and return them as a list of tensors.
        """
        result = []
        for i in range(sequences.shape[0]):
            # Remove padding from the sequence
            # Find the index of first eos token, keep the first eos token and remove everything after
            # If no eos token, keep the whole sequence
            eos_mask = sequences[i] == self.tokenizer.eos_token_id
            eos_idx = torch.nonzero(eos_mask)
            if eos_idx.shape[0] > 0:
                eos_idx = eos_idx[0].item()
                result.append(sequences[i][:eos_idx + 1])
            else:
                result.append(sequences[i])
        return result

    def _generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int,
        do_sample: bool,
        # grammar_str: str | None,
        constrain: bool,
        prefix_ids: torch.Tensor | None,
        num_return_sequences: int = 1,
        # temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        gcd_logits_processor = None
        if constrain:
            # grammar_constraint = IncrementalGrammarConstraint(grammar_str, "root", self.tokenizer)
            self.grammar_constraint.reset()
            if prefix_ids is not None:
                # TODO: can we do this always?
                gcd_logits_processor = GrammarConstrainedLogitsProcessor(self.grammar_constraint, len(input_ids[0]))
            else:
                gcd_logits_processor = GrammarConstrainedLogitsProcessor(self.grammar_constraint)

        logits_processor_list = []
        logits_processor_list.append(InfNanRemoveLogitsProcessor())
        if gcd_logits_processor is not None:
            logits_processor_list = [gcd_logits_processor] + logits_processor_list
        logits_processor_list = LogitsProcessorList(logits_processor_list)

        input_prefix_ids = input_ids
        if prefix_ids is not None:
            input_prefix_ids = torch.cat([input_ids, prefix_ids], dim=-1)

        output = self.model.generate(
            input_prefix_ids,
            generation_config=generation_config,
            tokenizer=self.tokenizer,
            # logits_processor=[gcd_logits_processor] if gcd_logits_processor else None,
            logits_processor=logits_processor_list,
        )

        output_ids = output.sequences
        output_ids = output_ids[:, input_prefix_ids.shape[1]:]
        output_scores = torch.stack(output.scores, dim=1)
        # Check that the length of the output and the scores match
        assert output_ids.shape[1] == output_scores.shape[1]

        return output_ids, output_scores

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        # grammar_str: str | None = None, 
        constrain: bool = False,
        prefix: str | None = None,
    ):
        prompt = self._format_prompt(prompt)
        input_str = prompt
        if prefix:
            input_str += prefix

        # We do this weird mangling to make sure the prefix is tokenized the same way as it would be produced
        # Ideally we could just encode the prefix but that does not work directly with some tokenizers
        input_ids = self.tokenizer.encode(input_str, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        prefix_ids = None
        if prefix:
            prefix_ids = input_ids[:, prompt_ids.shape[1]:]
        
        output_ids, _ = self._generate(prompt_ids, max_new_tokens, do_sample, constrain, prefix_ids)
        output_str = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        return output_str

    def _get_seq_logprob_from_scores(self, scores: torch.Tensor, query_ids: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of the sequences in `query_ids` given the `scores`.
        `scores` has shape (batch_size, seq_len, vocab_size).
        `query_ids` has shape (batch_size, seq_len).
        Result has shape (batch_size,).
        """

        assert scores.shape[0] == query_ids.shape[0], "Batch sizes must match"
        assert scores.shape[1] == query_ids.shape[1], "Sequence lengths must match"
    
        # Apply log_softmax to get log-probabilities
        logprobs = torch.log_softmax(scores, dim=-1)
    
        batch_size, seq_len = query_ids.shape
    
        # Initialize result tensor
        result = torch.zeros(batch_size, device=scores.device)
    
        # Process each sequence in the batch
        for i in range(batch_size):
            # Get logprobs for this sequence's tokens
            seq_token_logprobs = logprobs[i, torch.arange(seq_len), query_ids[i]]

            # Find the first EOS token's position (if any)
            eos_mask = query_ids[i] == self.tokenizer.eos_token_id
            # print(eos_mask)
            # eos_positions = torch.nonzero(eos_mask)[0]
            eos_positions = torch.nonzero(eos_mask)
            # print(eos_positions.shape)
            # eos_positions = torch.where(eos_mask)[0]

            # if len(eos_positions) > 0:
            if eos_positions.shape[0] > 0:
                # Include up to and including the first EOS token
                first_eos_pos = eos_positions[0].item()
                # Sum only up to and including the first EOS token
                # print(f"First EOS position: {first_eos_pos}")
                result[i] = seq_token_logprobs[:first_eos_pos + 1].sum()
            else:
                # No EOS token, sum all logprobs
                # print("No EOS token found")
                result[i] = seq_token_logprobs.sum()
    
        return result

    def _get_generation_scores(
        self,
        prompt_ids: torch.Tensor,
        query_ids: torch.Tensor,
        constrain: bool = False,
        prefix_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return the model generation scores at each step of the query using a single forward pass.
        More efficient than _get_generation_scores for longer sequences.
        """
        # If prompt_ids has batch size of 1, duplicate it to match the batch size of query_ids
        if prompt_ids.shape[0] == 1:
            prompt_ids = prompt_ids.repeat(query_ids.shape[0], 1)

        # Concatenate prompt_ids and query_ids to get the full sequence
        if prefix_ids is not None:
            input_ids = torch.cat([prompt_ids, prefix_ids, query_ids], dim=-1)
            prompt_prefix_len = prompt_ids.shape[1] + prefix_ids.shape[1]
        else:
            input_ids = torch.cat([prompt_ids, query_ids], dim=-1)
            prompt_prefix_len = prompt_ids.shape[1]
        
        # Single forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, return_dict=True)
            
        # Extract the logits for each position
        all_logits = outputs.logits
        # Get scores for each position corresponding to query_ids
        scores = all_logits[:, prompt_prefix_len-1:prompt_prefix_len+query_ids.shape[1]-1, :]
        
        # Apply grammar constraint if needed
        if constrain:
            self.grammar_constraint.reset()
            logits_processor = GrammarConstrainedLogitsProcessor(self.grammar_constraint, 0)
            modified_scores = []
            
            # Initialize with prompt_ids for the first step
            if prefix_ids is not None:
                current_ids = prefix_ids.clone()
            else:
                # Initialize current_ids, as an empty tensor with shape (batch_size, 0)
                # current_ids = torch.empty((1, 0), dtype=torch.long).to(self.model.device)
                current_ids = torch.empty((query_ids.shape[0], 0), dtype=torch.long).to(self.model.device)
            
            for i in range(query_ids.shape[1]):
                # Apply grammar constraint for this position
                current_step_logits = scores[:, i, :].clone()
                constrained_logits = logits_processor(current_ids, current_step_logits)
                modified_scores.append(constrained_logits)
                
                # Update current_ids for the next step
                current_ids = torch.cat([current_ids, query_ids[:, i:i+1]], dim=-1)
            
            # Stack along sequence dimension
            modified_scores = torch.stack(modified_scores, dim=1)
            scores = modified_scores
        
        assert scores.shape[1] == query_ids.shape[1]
        
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return scores

    def _get_seq_logprob(
        self, 
        prompt_ids: torch.Tensor,
        query_ids: torch.Tensor,
        # grammar_str: str | None = None,
        constrain: bool = False,
        prefix_ids: torch.Tensor | None = None,     
    ) -> torch.Tensor:
        """
        Get the log probability of a sequence given the model.
        """
        scores = self._get_generation_scores(prompt_ids, query_ids, constrain, prefix_ids)
        logprob = self._get_seq_logprob_from_scores(scores, query_ids)
        return logprob
    
    def _resample_idx_distribution(
        self,
        propose_style: str,
        current_ids: torch.Tensor,
        current_scores: torch.Tensor,
    ) -> torch.Tensor:
        if propose_style == "restart":
            # for resampling, we always resample from the beginning
            resample_distr = torch.zeros(len(current_ids[0]), dtype=torch.float32)
            resample_distr[0] = 1.0
            resample_distr = torch.unsqueeze(resample_distr, 0)
        elif propose_style == "prefix":
            # for prefix sampling, the distribution is uniform
            # resample_distr = [1 / len(current_ids[0]) for _ in range(len(current_ids[0]))]
            resample_distr = torch.ones(len(current_ids[0])) / len(current_ids[0])
            resample_distr = torch.unsqueeze(resample_distr, 0)
        elif propose_style == "priority":
            # for priority sampling, the distribution is proportional to the entropy
            # current_logprobs.shape = (1, current_ids.shape[1], vocab_size)
            current_logprobs = torch.log_softmax(current_scores, dim=-1)
            # Create a mask for non-inf logprobs to avoid NaN values
            mask = torch.isfinite(current_logprobs)
            probs = torch.exp(current_logprobs)
            # Zero out any non-finite values in the probs and calculate entropy properly
            masked_contribution = torch.where(mask, probs * current_logprobs, torch.zeros_like(probs))
            current_entropies = -torch.sum(masked_contribution, dim=-1)

            # get a probability for each index that is proportional to the entropy
            # we do -1 in order to zero out entropies of 0
            # TODO: is there a more principled way to do this?
            resample_distr = torch.exp(current_entropies) - 1
            resample_distr = resample_distr / torch.sum(resample_distr)
            # print(resample_distr)
            # print(resample_distr.shape)
        else:
            raise ValueError(f"Unknown proposal style: {propose_style}")
        # assert resample_distr.shape[0] == len(current_ids[0])
        assert resample_distr.shape == current_ids.shape
        assert torch.allclose(resample_distr.sum(), torch.tensor(1.0))
        return resample_distr
    
    def _propose_next_sequence_logprob(
        self,
        current_ids: torch.Tensor,
        current_scores: torch.Tensor,
        next_ids: torch.Tensor,
        next_scores: torch.Tensor,
        propose_style: str,
        # resample_idx_distr: torch.Tensor,
    ) -> float:
        resample_idx_distr = self._resample_idx_distribution(
            propose_style, current_ids, current_scores
        )

        # get the longest common prefix between the proposal and the current
        lcp_idx = 0
        for i, (p, c) in enumerate(zip(next_ids[0], current_ids[0])):
            if p == c:
                lcp_idx += 1
            else:
                break
        max_resample_idx = lcp_idx + 1
        max_resample_idx = min(max_resample_idx, len(current_ids[0]))

        # compute the probability of the proposal
        proposal_logprob = -np.inf
        for i in range(max_resample_idx):
            # Get probability of selecting this index
            # idx_resample_logprob = -np.log(len(current_ids[0]))
            idx_resample_prob = resample_idx_distr[0][i].item()
            if idx_resample_prob == 0:
                continue
            idx_resample_logprob = np.log(idx_resample_prob)
            
            # prefix_ids = current_ids[:, :i]
            suffix_ids = next_ids[:, i:]
            suffix_scores = next_scores[:, i:]

            # Get log probability in one call
            # suffix_logprob = self._get_seq_logprob(prompt_ids, suffix_ids, constrain, prefix_ids)
            suffix_logprob = self._get_seq_logprob_from_scores(suffix_scores, suffix_ids).item()
            
            # Add to total probability
            # proposal_prob += idx_resample_prob * np.exp(suffix_logprob)
            proposal_logprob = np.logaddexp(proposal_logprob, idx_resample_logprob + suffix_logprob)

        return proposal_logprob
    
    def _propose_next_sequence(
        self,
        prompt_ids: torch.Tensor,
        current_ids: torch.Tensor,
        max_new_tokens: int,
        constrain: bool,
        current_scores: torch.Tensor | None,
        propose_style: str,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        assert current_ids.shape[0] == 1
        if propose_style not in ["prefix", "priority", "restart"]:
            raise ValueError(f"Unknown proposal style: {propose_style}")
        
        if current_scores is None:
            current_scores = self._get_generation_scores(prompt_ids, current_ids, constrain)

        resample_idx_distr = self._resample_idx_distribution(
            propose_style, current_ids, current_scores
        ) 
        # print(resample_idx_distr.tolist())

        resample_idx = np.random.choice(len(current_ids[0]), p=resample_idx_distr[0].cpu().numpy())
        # print(resample_idx)
        print(f"Resample idx: {resample_idx}")

        # get the corresponding prefix from current tokens
        prefix_ids = current_ids[:, :resample_idx]
        prefix_scores = current_scores[:, :resample_idx]

        # resample from the prefix, using gcd
        resample_ids, resample_scores = self._generate(
            prompt_ids,
            max_new_tokens,
            do_sample=True,
            constrain=constrain,
            prefix_ids=prefix_ids,
        )
        # print([self.tokenizer.decode(token_id) for token_id in resample_ids[0]])

        next_ids = torch.cat([prefix_ids, resample_ids], dim=-1)
        next_scores = torch.cat([prefix_scores, resample_scores], dim=1)
        # print([self.tokenizer.decode(token_id) for token_id in next_ids[0]])

        proposal_logprob = self._propose_next_sequence_logprob(
            current_ids=current_ids,
            current_scores=current_scores,
            next_ids=next_ids,
            next_scores=next_scores,
            # resample_idx_distr=resample_idx_distr,
            propose_style=propose_style,
        )

        return next_ids, next_scores, proposal_logprob