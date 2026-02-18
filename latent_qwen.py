
import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

class LatentQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_latent_thoughts = kwargs.pop('num_latent_thoughts', getattr(config, 'num_latent_thoughts', 0))
        self.think_token_id = None
        
        # Initialize embedding shortcut for easy access (used in coconut logic)
        self.embedding = self.model.embed_tokens

    def set_special_token_ids(self, think_token_id):
        self.think_token_id = think_token_id

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
        **kwargs,
    ):
        # Fallback if no think token or 0 latents
        if self.think_token_id is None or self.num_latent_thoughts == 0:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        # 1. EXPAND INPUTS: Insert latent tokens if they are not already there?
        # The user wants "parameter that controls how many steps". 
        # We assume the input has ONE <think> and we expand it to N.
        # This is strictly necessary because Coconut logic iterates over N latent tokens in the input.
        
        # Check if we need to expand.
        # If we are training, input_ids is (B, L).
        # We find <think>.
        
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # We need to construct new_input_ids, new_mask, new_labels
        # This is tricky in batch if <think> position varies. 
        # We'll assume consistency or handle row-by-row like before but construct tensors.
        
        expanded_input_ids_list = []
        expanded_mask_list = []
        expanded_labels_list = []
        
        has_think = False
        
        for i in range(batch_size):
            row_ids = input_ids[i]
            row_mask = attention_mask[i] if attention_mask is not None else torch.ones_like(row_ids)
            row_labels = labels[i] if labels is not None else torch.full_like(row_ids, -100) # Dummy
            
            idx = (row_ids == self.think_token_id).nonzero()
            if len(idx) == 0:
                expanded_input_ids_list.append(row_ids)
                expanded_mask_list.append(row_mask)
                expanded_labels_list.append(row_labels)
                continue
                
            has_think = True
            idx = idx[0].item() # Index of <think>
            
            # Structure: Prefix + <think>(original) + [Latent]*N + Suffix
            # WAIT. Coconut replaces the embedding of the latent token with the PREVIOUS hidden state.
            # So if we have `A <think> B`. 
            # We want `A <think> [thought] [thought] ... B`.
            # First <think>: Standard input. Produces hidden state h0.
            # First [thought]: Input embedding replaced by h0. Produces h1.
            # Second [thought]: Input embedding replaced by h1. Produces h2.
            # ...
            # Suffix B: Input embedding standard.
            
            # So we need to insert `num_latent_thoughts` tokens.
            # What token ID? `think_token_id` is fine as placeholder.
            
            prefix = row_ids[:idx+1] # Includes <think>
            suffix = row_ids[idx+1:]
            
            latents = torch.full((self.num_latent_thoughts,), self.think_token_id, device=device, dtype=row_ids.dtype)
            
            new_row = torch.cat([prefix, latents, suffix])
            expanded_input_ids_list.append(new_row)
            
            # Mask
            prefix_mask = row_mask[:idx+1]
            suffix_mask = row_mask[idx+1:]
            latent_mask = torch.ones((self.num_latent_thoughts,), device=device, dtype=row_mask.dtype)
            expanded_mask_list.append(torch.cat([prefix_mask, latent_mask, suffix_mask]))
            
            # Labels
            prefix_labels = row_labels[:idx+1]
            suffix_labels = row_labels[idx+1:]
            latent_labels = torch.full((self.num_latent_thoughts,), -100, device=device, dtype=row_labels.dtype)
            expanded_labels_list.append(torch.cat([prefix_labels, latent_labels, suffix_labels]))

        if not has_think:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        # Pad the batch
        # Simple padding to max length
        max_len = max(len(x) for x in expanded_input_ids_list)
        pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        
        padded_ids = torch.full((batch_size, max_len), pad_id, device=device, dtype=input_ids.dtype)
        padded_mask = torch.zeros((batch_size, max_len), device=device, dtype=input_ids.dtype) # 0 for padding? usually 0 is masked
        # Transformers attention_mask: 1 for keep, 0 for mask.
        
        padded_labels = torch.full((batch_size, max_len), -100, device=device, dtype=input_ids.dtype)
        
        # position_ids?
        # We should generate them or let model generate.
        
        for i in range(batch_size):
            l = len(expanded_input_ids_list[i])
            padded_ids[i, :l] = expanded_input_ids_list[i]
            padded_mask[i, :l] = expanded_mask_list[i]
            padded_labels[i, :l] = expanded_labels_list[i]
            
        input_ids = padded_ids
        attention_mask = padded_mask
        labels = padded_labels
        
        # Generate full position_ids
        # (B, L_expanded)
        # Note: If batch items have different padding start, we ideally mask/adjust position_ids?
        # Standard transformers: 0..L-1, masked tokens are ignored. Left-padding handled by user.
        # We assume standard right-padding or handle simple range.
        position_ids = torch.arange(input_ids.shape[1], device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        
        # --- COCONUT LOGIC START ---
        
        # Identify latent indices
        latent_indices_list = []
        for i in range(batch_size):
            indices = (input_ids[i] == self.think_token_id).nonzero().squeeze(-1)
            # indices[0] is original. indices[1:] are latents.
            if len(indices) > 1:
                latent_indices_list.append(indices[1:])
            else:
                latent_indices_list.append(torch.tensor([], device=device, dtype=torch.long))

        # Max number of passes needed
        max_latents = max(len(l) for l in latent_indices_list)
        
        # Inputs Embeds
        inputs_embeds = self.embedding(input_ids)
        
        kv_cache = None
        current_pos = 0 # Track compute head
        
        latent_positions = [t.tolist() for t in latent_indices_list]
        max_n_latents = max(len(l) for l in latent_positions)
        
        current_latent_idx = [0] * batch_size
        
        next_compute_start = 0
        tokens_seq_len = input_ids.shape[1]
        
        logits_list = []
        outputs = None # Store last output for hidden states
        
        while next_compute_start < tokens_seq_len:
            # Determine next stop
            stops = []
            for b in range(batch_size):
                latents = latent_positions[b]
                idx = current_latent_idx[b]
                if idx < len(latents):
                    stops.append(latents[idx])
                else:
                    stops.append(tokens_seq_len)
            
            next_stop = min(stops)
            
            if to_update_indices := [b for b, latents in enumerate(latent_positions) 
                                     if current_latent_idx[b] < len(latents) 
                                     and latents[current_latent_idx[b]] == next_compute_start]:
                
                # Update embeddings with previous hidden state
                # Need hidden states from previous chunk.
                # `outputs` is from previous iteration.
                if outputs is not None:
                    # hidden_states reported by model is usually tuple of (L, B, S, H) or (B, S, H) depending.
                    # Qwen2ForCausalLM returns CausalLMOutputWithPast. hidden_states is tuple.
                    # We want Last Layer.
                    last_layer_hidden = outputs.hidden_states[-1] # (B, S_chunk, H)
                    
                    for b in to_update_indices:
                        h = last_layer_hidden[b, -1, :]
                        inputs_embeds[b, next_compute_start, :] = h
                        current_latent_idx[b] += 1
                
                next_stop = next_compute_start + 1
            
            if next_stop <= next_compute_start:
                 next_stop = next_compute_start + 1
                 
            # Run Model for range
            chunk_embeds = inputs_embeds[:, next_compute_start:next_stop, :]
            chunk_mask = attention_mask[:, :next_stop]
            chunk_pos_ids = position_ids[:, next_compute_start:next_stop]
            
            model_kwargs = {
                "output_hidden_states": True,
                "return_dict": True
            }
            model_kwargs.update(kwargs) # user args
            model_kwargs["use_cache"] = True # Force True!
            model_kwargs.pop("past_key_values", None)
            model_kwargs.pop("inputs_embeds", None)
            model_kwargs.pop("attention_mask", None)
            model_kwargs.pop("labels", None)
            model_kwargs.pop("position_ids", None)
            model_kwargs.pop("cache_position", None)
            model_kwargs.pop("num_logits_to_keep", None)
            
            # Convert DynamicCache to legacy tuple to avoid Accelerate/AMP issues
            # REMOVED: Newer transformers require the object API (get_seq_length)
            past_input = kv_cache
            # if hasattr(kv_cache, "to_legacy_cache"):
            #     past_input = kv_cache.to_legacy_cache()

            # DEBUG prints removed
            
            outputs = super().forward(
                inputs_embeds=chunk_embeds,
                attention_mask=chunk_mask,
                position_ids=chunk_pos_ids,
                past_key_values=past_input,
                **model_kwargs
            )
            
            logits_list.append(outputs.logits)
            kv_cache = outputs.past_key_values
            
            next_compute_start = next_stop

        # Concatenate logits
        full_logits = torch.cat(logits_list, dim=1) # (B, L, V)
        
        # Loss computation?
        # If labels are provided, super() usually computes loss, but we bypassed it.
        loss = None
        if labels is not None:
            # Shift
            shift_logits = full_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=full_logits,
            past_key_values=kv_cache,
            hidden_states=outputs.hidden_states, # Only last chunk?
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        # We need to handle the mismatch between `input_ids` (visible) and `past_key_values` (visible + latents).
        # Check if we have past_key_values and if they are longer than input_ids + standard attention mask.
        
        # Standard preparation first
        model_inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        
        # If we have a past, check its length.
        if past_key_values is not None:
            # past_key_values is typically a tuple of tuples. ((k, v), ...)
            # We just need the length of the sequence in the cache.
            # Qwen/HF implementations vary, but usually `past_key_values[0][0].shape[2]` is seq_len.
            
            past_length = 0
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            else:
                 # Legacy tuple cache
                 past_length = past_key_values[0][0].shape[-2]
            
            # The `attention_mask` in `model_inputs` is usually constructed based on `input_ids` shape.
            # If `past_length` > mask length, we need to extend the mask.
            
            if "attention_mask" in model_inputs:
                mask = model_inputs["attention_mask"]
                if mask.shape[1] < past_length + input_ids.shape[1]: 
                     # input_ids is usually length 1 during generation step (except first).
                     # Wait, `prepare_inputs` is called. `input_ids` passed here is usually just the *new* tokens if past is present.
                     # But `attention_mask` passed in kwargs is usually the full mask?
                     # Let's check `attention_mask` shape.
                     
                     # HF `generate` maintains `attention_mask`. 
                     # If we injected tokens secretly, the mask maintained by `generate` is too short.
                     # We must pad it with 1s to the left (prefix) or ensure logic holds.
                     
                     diff = (past_length + input_ids.shape[1]) - mask.shape[1]
                     if diff > 0:
                         # We append 1s to the mask? 
                         # Actually the latent tokens are 'in the past', so they should be 1s.
                         # We prepend/append to match the `past_key_values` structure.
                         # Usually mask aligns with `past + input`.
                         
                         ones = torch.ones((mask.shape[0], diff), device=mask.device, dtype=mask.dtype)
                         model_inputs["attention_mask"] = torch.cat([ones, mask], dim=1)
        
        return model_inputs

    def generate(self, input_ids=None, **kwargs):
        # Logic: 
        # 1. Check if we are doing latent generation.
        #    Condition: input_ids ends with <think>.
        # 2. If so, forward pass will act normally (expand and compute latents).
        # 3. We must ensure the NEXT generated token is </think>.
        
        if input_ids is None:
             return super().generate(**kwargs)

        # Retrieve think_token_id
        think_id = self.think_token_id
        
        should_add_processor = False
        if think_id is not None:
            # Check last token of inputs
            # input_ids is (B, L)
            last_tokens = input_ids[:, -1]
            if (last_tokens == think_id).any():
                should_add_processor = True

        if should_add_processor:
            # We add a LogitsProcessor that forces </think> immediately after the latent expansion.
            
            from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

            class LatentControlLogitsProcessor(LogitsProcessor):
                def __init__(self, think_id, close_think_id):
                    self.think_id = think_id
                    self.close_think_id = close_think_id
                
                def __call__(self, input_ids, scores):
                    # input_ids: (B, L)
                    last = input_ids[:, -1]
                    
                    # Case 1: Last visible token was <think>.
                    # This means we just processed the prompt + latents (in forward).
                    # We must close the thought.
                    mask_think = (last == self.think_id)
                    if mask_think.any():
                        # Force </think>
                        scores[mask_think, :] = -float('inf')
                        scores[mask_think, self.close_think_id] = 0
                    
                    # We do NOT force <answer> after </think>.
                    # The model triggers generation "as usual".
                    return scores

            # Create processor
            close_think_id = getattr(self, 'close_think_id', getattr(self, 'close_think_token_id', None))
            
            if close_think_id is not None:
                proc = LatentControlLogitsProcessor(think_id, close_think_id)
                lp = kwargs.get('logits_processor', LogitsProcessorList())
                lp.append(proc)
                kwargs['logits_processor'] = lp

        return super().generate(input_ids, **kwargs)
