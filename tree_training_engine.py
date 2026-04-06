
import torch
from transformers.cache_utils import DynamicCache
from typing import List, Optional, Tuple
import time

import torch.nn.functional as F
from math import ceil
from bisect import bisect_left, bisect_right

from vocab_parallel import gather_logprobs, gather_logprobs_entropy


def _time_now(sync_cuda: bool) -> float:
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _prof_add(profile: Optional[dict], key: str, delta: float):
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + float(delta)


def _prof_inc(profile: Optional[dict], key: str, value: int = 1):
    if profile is not None:
        profile[key] = profile.get(key, 0) + int(value)


def _get_forkpos(lens, lcp_lens, block_size: int) -> list:
    """
    Compute all fork positions that TreeTrainingEngine's stack must track.

    Fork positions are token indices where:
    1) Sequences diverge (longest common prefix boundaries)
    2) Block boundaries for long sequences to reduce memory usage

    Returns a sorted list of unique fork positions.
    """

    forkpos_list = []

    # 1. Fork positions induced by branching (LCP boundaries)
    for lcp in lcp_lens:
        if lcp > 0:
            forkpos_list.append(lcp - 1)

    # 2. Fork positions induced by block segmentation
    if block_size is not None:
        for i in range(len(lens)):
            start = 0 if i == len(lcp_lens) else lcp_lens[i]
            end = lens[i]

            pop_len = end - start
        
            n_blocks = ceil(pop_len / block_size)
            block_size_actual = ceil(pop_len / n_blocks)

            for b in range(n_blocks):
                pop_start = max(end - (b + 1) * block_size_actual, start)
                if pop_start > 0:
                    forkpos_list.append(pop_start - 1)

    forkpos_list = list(set(forkpos_list))
    forkpos_list.sort()

    return forkpos_list


class TreeTrainingEngine:
    """
    Engine for backward computation over sequences with shared prefixes.

    TreeTrainingEngine stores only necessary KV caches, logits at fork
    positions, log-probs, and entropy to efficiently compute gradients
    for multiple sequences while saving memory.

    Supports block-wise popping to reduce GPU memory peak.
    """
    def __init__(self, model_config, device, dtype: torch.dtype, max_seq_len: int, forward_only: bool = False):
        """
        Initialize the engine with model_config, device, dtype, and maximum sequence length.

        Buffers for tokens, logprobs, entropy and KV caches are preallocated 
        to max_seq_len.
        """
        self.model = None
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # ------------------------------------------------------------------------
        # Initialize static stack buffers
        # ------------------------------------------------------------------------
        self.cur_len = 0
        
        # Token buffer
        self.tokens        = torch.zeros((max_seq_len), device=self.device, dtype=torch.long)

        # Entropy buffer
        if not forward_only:
            self.entropy       = torch.zeros((max_seq_len), device=self.device, dtype=torch.float32)
            self.grad_entropy  = torch.zeros((max_seq_len), device=self.device, dtype=dtype)

        # Logprob buffer
        self.logprobs      = torch.zeros((max_seq_len), device=self.device, dtype=torch.float32)
        if not forward_only:
            self.grad_logprobs = torch.zeros((max_seq_len), device=self.device, dtype=dtype)

        # Fork position logits buffer (store logits only at fork positions, others are None)
        self.forkpos_list  = []                                                             # List of all fork positions
        self.forkpos_logits: List[Optional[torch.Tensor]]          = [None] * max_seq_len   # Logits at fork positions for computing logprobs
        if not forward_only:
            self.grad_forkpos_logits: List[Optional[torch.Tensor]] = [None] * max_seq_len   # Gradients of logits at fork positions

        # Attachments buffer
        self.attachs       = [] # List of sequences retained in the stack, including (attachments, length)

        # KV cache buffers
        self.n_layers = model_config.num_hidden_layers
        n_kv_heads = model_config.num_key_value_heads
        # Compatible with Qwen2.5 and Qwen3 series
        head_dim = model_config.head_dim if hasattr(model_config, 'head_dim') \
            else model_config.hidden_size // model_config.num_attention_heads

        # Stacked KV cache: [2, n_layers, 1, n_kv_heads, max_seq_len, head_dim]
        # dim 0: 0=keys, 1=values
        kv_stacked_shape = (2, self.n_layers, 1, n_kv_heads, max_seq_len, head_dim)
        self.kv_cache = torch.zeros(kv_stacked_shape, device=self.device, dtype=dtype)

        if not forward_only:
            self.grad_kv = torch.zeros(kv_stacked_shape, device=self.device, dtype=dtype)

        self.ret_logprobs = []
        self.last_profile = {}
    
    def get_forkpos(self, start: int, end: int) -> List[int]:
        """
        Yield fork positions within the interval [start, end).

        Uses binary search on precomputed forkpos_list.
        """

        left = bisect_left(self.forkpos_list, start)
        right = bisect_right(self.forkpos_list, end - 1)
        yield from self.forkpos_list[left:right]

    @torch.no_grad()
    def push_forward_only(
        self,
        new_tokens: torch.LongTensor,
        attach_list: List[Tuple[dict, int]],
    ):
        """
        Push new tokens into the stack with their attachments.

        Builds cache (KV, logprobs) up to cache_len.
        Updates logprobs for the previous token.

        Used in inference mode only.
        """
        
        B = new_tokens.numel()
        assert self.cur_len + B <= self.max_seq_len, (
            f"Exceeds max_seq_len: cur_len={self.cur_len}, new_tokens={B}, max={self.max_seq_len}"
        )

        start, end = self.cur_len, self.cur_len + B

        # -------------------------------------------------------------
        # 1. Build prefix cache from existing KV
        # -------------------------------------------------------------
        prefix_cache = DynamicCache()
        for l in range(self.n_layers):
            prefix_cache.update(
                self.kv_cache[0, l, :, :, :start, :],
                self.kv_cache[1, l, :, :, :start, :],
                layer_idx=l,
            )

        # -------------------------------------------------------------
        # 2. Forward
        # -------------------------------------------------------------
        out = self.model(
            new_tokens.unsqueeze(0),
            past_key_values=prefix_cache,
            use_cache=True,
        )
        
        # Compute logprobs and entropy for new tokens
        logits = out.logits  # [1, B, vocab]
        logprobs = gather_logprobs(
            logits=logits,
            labels=new_tokens[1:].unsqueeze(0),
        )

        # -------------------------------------------------------------
        # 3. Write tokens, computed logprobs, and KV cache into stack
        # -------------------------------------------------------------

        # Write tokens into stack
        self.tokens[start:end] = new_tokens

        # Write logprobs into stack
        self.logprobs[start : end-1] = logprobs.squeeze(0)
        # Fill the logprob of the first token using self.forkpos_logits[start]
        if start > 0:   
            pre_logits = self.forkpos_logits[start-1].float()
            first_token = new_tokens[0].item()
            pre_logprob = F.log_softmax(pre_logits, dim=-1)[first_token].item()
            self.logprobs[start-1] = pre_logprob

        # Write KV cache into stack
        new_cache = out.past_key_values 
        for l, layer in enumerate(new_cache.layers):
            self.kv_cache[0, l, :, :, start:end, :] = layer.keys[:, :, start:end, :]
            self.kv_cache[1, l, :, :, start:end, :] = layer.values[:, :, start:end, :]

        # Write logits into stack (fork positions only)
        forkpos_slice = self.get_forkpos(start, end)
        for i in forkpos_slice:
            self.forkpos_logits[i] = logits[0, i - start].detach().clone()

        # -------------------------------------------------------------
        # 4. Store logprobs for sequences ending in attach_list
        # -------------------------------------------------------------
        for attachment, length in attach_list:
            seq_id = attachment['_sequence_batch_id']
            logprobs = self.logprobs[:length-1]
            self.returns[seq_id] = logprobs.clone()

        self.cur_len += B

    def build_cache(self, start: int, end: int, profile: Optional[dict] = None, profile_cuda_sync: bool = True):
        """
        Build KV cache, logprobs and entropy for tokens in [start, end).
        Uses the existing prefix cache [0, start).
        """
        
        if profile is not None:
            _prof_inc(profile, "build_cache_calls", 1)
            _prof_inc(profile, "build_cache_tokens", end - start)
            t_build = _time_now(profile_cuda_sync)

        # Build prefix cache from existing KV
        prefix_cache = DynamicCache()
        for l in range(self.n_layers):
            prefix_cache.update(
                self.kv_cache[0, l, :, :, :start, :],
                self.kv_cache[1, l, :, :, :start, :],
                layer_idx=l,
            )

        # Forward pass to compute new KV
        out = self.model(
            self.tokens[start:end].unsqueeze(0),
            past_key_values=prefix_cache,
            use_cache=True,
        )
        
        # Compute logprobs & entropy for new tokens
        logits = out.logits  # [1, B, vocab]
        logprobs, entropy = gather_logprobs_entropy(
            logits=logits,
            labels=self.tokens[start+1:end].unsqueeze(0),
        )
        self.logprobs[start:end-1] = logprobs.squeeze(0)
        self.entropy[start:end] = entropy.squeeze(0)

        # Write new KV cache into stack
        new_cache = out.past_key_values 
        for l, layer in enumerate(new_cache.layers):
            self.kv_cache[0, l, :, :, start:end, :] = layer.keys[:, :, start:end, :]
            self.kv_cache[1, l, :, :, start:end, :] = layer.values[:, :, start:end, :]

        # Write logits into stack (fork positions only)
        forkpos_slice = self.get_forkpos(start, end)
        for i in forkpos_slice:
            self.forkpos_logits[i] = logits[0, i - start].detach().clone()

        if profile is not None:
            _prof_add(profile, "build_cache_time", _time_now(profile_cuda_sync) - t_build)

    @torch.no_grad()
    def push(
        self,
        new_tokens: torch.LongTensor,
        attachs: List[Tuple[dict, int]],
        cache_len: int,
        profile: Optional[dict] = None,
        profile_cuda_sync: bool = True,
    ):
        """
        Push new tokens into the stack with their attachments.

        Builds cache (KV, logprobs, entropy) up to cache_len.
        Updates logprobs for the previous token.
        """
        
        B = new_tokens.numel()
        assert self.cur_len + B <= self.max_seq_len, (
            f"Exceeds max_seq_len: cur_len={self.cur_len}, new_tokens={B}, max={self.max_seq_len}"
        )

        start, end = self.cur_len, self.cur_len + B
        if profile is not None:
            _prof_inc(profile, "push_calls", 1)
            _prof_inc(profile, "push_tokens", B)
            t_push = _time_now(profile_cuda_sync)

        # Add attachments
        for attachment, length in attachs:
            self.attachs.append((attachment, length))

        # Write tokens
        self.tokens[start:end] = new_tokens

        # Build prefix cache (KV & logprobs/entropy) if needed
        if start < cache_len:
            self.build_cache(start, cache_len, profile=profile, profile_cuda_sync=profile_cuda_sync)

        # 修改上一个 token 的 logprob
        if start > 0:
            pre_logits = self.forkpos_logits[start-1].float()
            first_token = new_tokens[0].item()
            pre_logprob = F.log_softmax(pre_logits, dim=-1)[first_token].item()
            self.logprobs[start-1] = pre_logprob

        self.cur_len = end
        if profile is not None:
            _prof_add(profile, "push_time", _time_now(profile_cuda_sync) - t_push)

    def pop(
        self,
        start: int,
        loss_fn,
        profile: Optional[dict] = None,
        profile_cuda_sync: bool = True,
    ) -> float:
        """
        Pop tokens from position `start` to the current end.

        Computes gradients for the popped tokens and accumulates them
        into the stack's KV, logprobs, entropy, and fork position logits buffers.

        Args:
            start: The starting token index to pop from.
            loss_fn: Callable that computes the loss for a sequence segment.

        Returns:
            The total loss computed over sequences ending within the popped segment.
        """
        assert 0 <= start < self.cur_len, f"Invalid start={start}, cur_len={self.cur_len}"

        end = self.cur_len
        B = end - start
        if profile is not None:
            _prof_inc(profile, "pop_calls", 1)
            _prof_inc(profile, "pop_tokens", B)
            t_pop = _time_now(profile_cuda_sync)

        tokens_to_pop = self.tokens[start:end]

        # ---------------------------------------------------------------------------------
        # 1. Gather prefix KV (stacked, 2 tensors instead of 2*n_layers)
        # ---------------------------------------------------------------------------------
        if profile is not None:
            t_prefix = _time_now(profile_cuda_sync)
        prefix_keys = self.kv_cache[0, :, :, :, :start, :].detach().requires_grad_(True)
        prefix_values = self.kv_cache[1, :, :, :, :start, :].detach().requires_grad_(True)
        prefix_cache = DynamicCache()
        for l in range(self.n_layers):
            prefix_cache.update(prefix_keys[l], prefix_values[l], layer_idx=l)
        if profile is not None:
            _prof_add(profile, "pop_prefix_prep_time", _time_now(profile_cuda_sync) - t_prefix)

        # ---------------------------------------------------------------------------------
        # 2. Forward pass on tokens_to_pop (builds computation graph)
        # ---------------------------------------------------------------------------------
        if profile is not None:
            t_fwd = _time_now(profile_cuda_sync)
        out = self.model(
            tokens_to_pop.unsqueeze(0), past_key_values=prefix_cache, use_cache=True
        )
        if profile is not None:
            _prof_add(profile, "pop_forward_graph_time", _time_now(profile_cuda_sync) - t_fwd)
        
        logits = out.logits
        block_cache = out.past_key_values

        # ---------------------------------------------------------------------------------
        # 3. Compute suffix logprobs & entropy
        # ---------------------------------------------------------------------------------
        if profile is not None:
            t_post_fwd = _time_now(profile_cuda_sync)
        suf_logprobs, suf_entropy = gather_logprobs_entropy(
            logits=logits,
            labels=tokens_to_pop[1:].unsqueeze(0)
        )
        suf_entropy = suf_entropy.squeeze(0)
        suf_logprobs = suf_logprobs.squeeze(0)
        
        # Compute logprob for connection to previous token if exists
        if start > 0:
            mid_logits = self.forkpos_logits[start-1].float().detach().requires_grad_(True)
            mid_label = self.tokens[start].item()
            mid_logprob = F.log_softmax(mid_logits, dim=-1)[mid_label].unsqueeze(0)
        if profile is not None:
            _prof_add(profile, "pop_post_forward_time", _time_now(profile_cuda_sync) - t_post_fwd)

        # ---------------------------------------------------------------------------------
        # 4. Compute loss for sequences ending in this block
        # ---------------------------------------------------------------------------------

        # Gather attachs for sequences ending in this block
        if profile is not None:
            t_stack = _time_now(profile_cuda_sync)
        attachs_in_block = [(att, length) for att, length in self.attachs if start < length <= end]
        if profile is not None:
            _prof_inc(profile, "pop_attachs_in_block_total", len(attachs_in_block))
            _prof_add(profile, "pop_stack_filter_time", _time_now(profile_cuda_sync) - t_stack)

        if attachs_in_block:
            # Concatenate full logprobs and entropy, with requires_grad=True
            if start > 0:
                pre_entropy = self.entropy[:start].detach().requires_grad_(True)
                entropys = torch.cat([pre_entropy, suf_entropy], dim=0)
                if start > 1:
                    pre_logprobs = self.logprobs[:start-1].detach().requires_grad_(True)
                    logprobs = torch.cat([pre_logprobs, mid_logprob, suf_logprobs], dim=0)
                else:
                    logprobs = torch.cat([mid_logprob, suf_logprobs], dim=0)
            else:
                entropys = suf_entropy
                logprobs = suf_logprobs

            # Compute loss
            if profile is not None:
                t_loss = _time_now(profile_cuda_sync)
            loss = 0.0
            for attachment, length in attachs_in_block:
                loss += loss_fn(logprobs[:length-1], entropys[:length], attachment)
            if profile is not None:
                _prof_add(profile, "pop_loss_build_time", _time_now(profile_cuda_sync) - t_loss)

        # ---------------------------------------------------------------------------------
        # 5. Backward with gradient injection from popped tokens 
        #    (to KV, logprobs, entropy, forkpos-logits)
        # ---------------------------------------------------------------------------------
        roots, grads = [], []

        # Loss gradient
        if attachs_in_block:
            roots.append(loss)
            grads.append(torch.tensor(1.0, device=self.device, dtype=loss.dtype))

        # KV gradients from popped tokens
        for l, layer in enumerate(block_cache.layers):
            k = layer.keys[:, :, start:end, :]
            v = layer.values[:, :, start:end, :]
            roots.extend([k, v])
            grads.extend(
                [
                    self.grad_kv[0, l, :, :, start:end, :],
                    self.grad_kv[1, l, :, :, start:end, :],
                ]
            )
        
        # Logprobs & entropy gradients from popped tokens
        roots.extend([suf_logprobs, suf_entropy])
        grads.extend([self.grad_logprobs[start:end-1], self.grad_entropy[start:end]])
        if start > 0:
            roots.append(mid_logprob)
            grad_mid_logprob = self.grad_logprobs[start-1].unsqueeze(0)
            grads.append(grad_mid_logprob)

        # Fork position logits gradients
        forkpos_slice = self.get_forkpos(start, end)
        for i in forkpos_slice:
            if self.grad_forkpos_logits[i] is not None:
                fork_logits = logits[0, i - start]
                roots.append(fork_logits)
                grads.append(self.grad_forkpos_logits[i])

        # roots: loss, (KV, logprobs, entropy, forkpos logits) in tokens_to_pop
        if profile is not None:
            t_bwd = _time_now(profile_cuda_sync)
        torch.autograd.backward(roots, grads)
        if profile is not None:
            _prof_add(profile, "pop_autograd_backward_time", _time_now(profile_cuda_sync) - t_bwd)
        
        # ---------------------------------------------------------------------------------
        # 6. Accumulate gradients to prefix cache (KV, logprobs, entropy, forkpos-logits)
        # ---------------------------------------------------------------------------------

        # gradients to prefix KV (vectorized: 2 ops instead of 2*n_layers)
        if profile is not None:
            t_acc = _time_now(profile_cuda_sync)
        if prefix_keys.grad is not None:
            self.grad_kv[0, :, :, :, :start, :] += prefix_keys.grad
        if prefix_values.grad is not None:
            self.grad_kv[1, :, :, :, :start, :] += prefix_values.grad

        if start > 0:
            # gradients to forkpos logits
            if mid_logits.grad is not None:
                if self.grad_forkpos_logits[start-1] is None:
                    self.grad_forkpos_logits[start-1] = mid_logits.grad.clone()
                else:
                    self.grad_forkpos_logits[start-1] += mid_logits.grad
            if attachs_in_block:
                # gradients to prefix logprobs & entropy
                if pre_entropy.grad is not None:
                    self.grad_entropy[:start] += pre_entropy.grad
                if start > 1 and pre_logprobs.grad is not None:
                    self.grad_logprobs[:start-1] += pre_logprobs.grad
        if profile is not None:
            _prof_add(profile, "pop_prefix_grad_accum_time", _time_now(profile_cuda_sync) - t_acc)

        # ---------------------------------------------------------------------------------
        # 7. Cleanup: truncate and clear buffers
        # ---------------------------------------------------------------------------------

        if profile is not None:
            t_clean = _time_now(profile_cuda_sync)
        self.attachs = [(att, length) for att, length in self.attachs if length <= start]

        self.grad_kv[:, :, :, :, start:end, :].zero_()

        self.grad_logprobs[0 if start==0 else start-1:end-1].zero_()
        self.grad_entropy[start:end].zero_()
        
        forkpos_slice = self.get_forkpos(start, end)
        for i in forkpos_slice:
            self.forkpos_logits[i] = None
            self.grad_forkpos_logits[i] = None

        self.cur_len = start
        if profile is not None:
            _prof_add(profile, "pop_cleanup_time", _time_now(profile_cuda_sync) - t_clean)
            _prof_add(profile, "pop_total_time", _time_now(profile_cuda_sync) - t_pop)

        return loss.item() if attachs_in_block else 0.0

    def pop_byblock(
        self,
        start: int,
        block_size: int,
        loss_fn,
        profile: Optional[dict] = None,
        profile_cuda_sync: bool = True,
    ) -> float:
        """
        Pop tokens from [start, cur_len) in blocks to reduce peak GPU memory usage.

        Tokens are popped in reverse block order, calling `pop()` on each block.

        Args:
            start: The starting token index to pop from.
            block_size: Maximum block size for each pop to control memory usage.
            loss_fn: Callable to compute loss for a sequence segment.

        Returns:
            Total loss over all popped blocks.
        """
        end = self.cur_len
        length = end - start
        n_blocks = ceil(length / block_size)
        block_size_actual = ceil(length / n_blocks)
        if profile is not None:
            _prof_inc(profile, "pop_byblock_calls", 1)
            _prof_inc(profile, "pop_byblock_blocks_total", n_blocks)

        loss = 0.0
        for b in range(n_blocks):
            pop_start = max(end - (b + 1) * block_size_actual, start)
            loss += self.pop(
                pop_start,
                loss_fn,
                profile=profile,
                profile_cuda_sync=profile_cuda_sync,
            )

        return loss

    @torch.no_grad()
    def forward(self, model, token_trie):
        """
        Perform backward pass over all sequences in a TokenTrie.
        Compute logprobs for each sequence.
        The sequence ID is identified by attachment['_sequence_batch_id'], which TokenTrie automatically adds.

        Args:
            token_trie: TokenTrie containing input sequences and attachs.

        Returns:
            List of logprob tensors for each sequence in the TokenTrie.
        """

        self.model = model
        self.returns = [None] * token_trie.n_sequences

        inputs, attach_lists, lcp_lens = token_trie.inputs, token_trie.attach_lists, token_trie.lcp_lens

        self.forkpos_list = _get_forkpos(None, lcp_lens, None)

        for i in range(len(inputs)):
            input_ids = inputs[i].to(self.device)
            attach_list = attach_lists[i]
            seq_len = input_ids.size(0)

            # Pop diverged branch from previous sequence
            if i > 0:
                self.cur_len = lcp_lens[i - 1]

            # Push new tokens
            new_tokens = input_ids[self.cur_len :]

            self.push_forward_only(new_tokens, attach_list)

        self.cur_len = 0
        self.forkpos_logits = [None] * self.max_seq_len  # Clear forkpos_logits

        return self.returns

    def backward(
        self,
        model,
        token_trie,
        loss_fn,
        block_size: int,
        cut_f1_tail: bool = True,
        profile: bool = False,
        profile_cuda_sync: bool = True,
    ) -> float:
        """
        Perform backward pass over all sequences in a TokenTrie.

        Processes sequences in lexicographic order, popping diverged
        branches (block-wise) and pushing new tokens.

        Args:
            token_trie: TokenTrie containing input sequences and attachs.
            block_size: Maximum block size for popping to control GPU memory.
            loss_fn: Callable to compute per-sequence loss.
            cut_f1_tail: Whether to cut the tail of the first forward.
        Returns:
            Total loss accumulated over all sequences.
        """

        self.model = model
        prof = {} if profile else None
        t_total = _time_now(profile_cuda_sync) if profile else None

        total_loss = 0.0

        inputs, attach_lists, lcp_lens = token_trie.inputs, token_trie.attach_lists, token_trie.lcp_lens

        # Precompute fork positions and block boundaries
        if profile:
            t_fork = _time_now(profile_cuda_sync)
        lens = [ids.size(0) for ids in inputs]
        self.forkpos_list = _get_forkpos(lens, lcp_lens, block_size)
        if profile:
            _prof_add(prof, "forkpos_precompute_time", _time_now(profile_cuda_sync) - t_fork)

        # Process each sequence
        for i in range(len(inputs)):
            if profile:
                _prof_inc(prof, "n_sequences", 1)
            input_ids = inputs[i].to(self.device)
            attach_list = attach_lists[i]
            seq_len = input_ids.size(0)

            # Pop diverged branch from previous sequence
            if i > 0:
                lcp = lcp_lens[i - 1]
                if lcp < self.cur_len:
                    total_loss += self.pop_byblock(
                        lcp,
                        block_size,
                        loss_fn,
                        profile=prof,
                        profile_cuda_sync=profile_cuda_sync,
                    )

            # Push new tokens
            new_tokens = input_ids[self.cur_len :]

            # Determine cache length to build (optimize for next pop)
            lcp_next = lcp_lens[i] if i < len(inputs) - 1 else 0
            B = new_tokens.numel()
            next_pop_len = self.cur_len + B - lcp_next

            if next_pop_len > block_size:
                n_blocks = ceil(next_pop_len / block_size)
                block_size_actual = ceil(next_pop_len / n_blocks)
                cache_len = max(self.cur_len + B - block_size_actual, lcp_next)
            else:
                cache_len = lcp_next

            if not cut_f1_tail:
                cache_len = self.cur_len + B

            self.push(
                new_tokens,
                attach_list,
                cache_len,
                profile=prof,
                profile_cuda_sync=profile_cuda_sync,
            )

        # Final pop for remaining tokens
        if self.cur_len > 0:
            total_loss += self.pop_byblock(
                0,
                block_size,
                loss_fn,
                profile=prof,
                profile_cuda_sync=profile_cuda_sync,
            )

        if profile:
            total = _time_now(profile_cuda_sync) - t_total
            _prof_add(prof, "tree_backward_total_time", total)
            # Time in Python-side scheduling / stack orchestration not captured by inner kernels.
            inner = (
                prof.get("push_time", 0.0)
                + prof.get("pop_total_time", 0.0)
                + prof.get("forkpos_precompute_time", 0.0)
            )
            prof["tree_backward_stack_other_time"] = max(0.0, total - inner)
            self.last_profile = prof
        else:
            self.last_profile = {}

        return total_loss