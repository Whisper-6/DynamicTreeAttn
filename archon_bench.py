from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from math import gcd
from math import ceil
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

# DynamicTreeAttn is commonly launched from its own directory. Make the adjacent
# AReaL package importable without requiring an editable installation.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from areal.api import FinetuneSpec, ParallelStrategy
from areal.api.cli_args import ArchonEngineConfig, MicroBatchSpec, TrainEngineConfig
from areal.experimental.engine.archon_engine import ArchonEngine
from areal.models.tree_attn.functional import (
    _gather_packed_tree_logprobs,
    _gather_packed_tree_logprobs_entropy,
)
from areal.models.tree_attn.module_archon import TreeAttentionMeta, TreeAttentionWrapper
from areal.models.tree_attn.tree import TrieNode
from areal.utils.functional.vocab_parallel import gather_logprobs, gather_logprobs_entropy

from data_parallel import LB_by_n_tokens, split_by_dfs_cost_limit
from token_trie import TokenTrie
from tree_time_model import TreeTimeModel


LossFn = Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor]

_PATCHED_TREE_ATTENTION = False
_PATCHED_TREE_ATTENTION_OPTIONS: dict[str, int] | None = None
_PATCHED_QWEN3_SPARSE_AC = False
DEFAULT_FLEX_BLOCK_SIZE = 128


def _validate_flex_block_size(flex_block_size: int) -> int:
    if flex_block_size <= 0:
        raise ValueError(f"flex_block_size must be positive, got {flex_block_size}.")
    return flex_block_size


def _flex_kernel_options(flex_block_size: int) -> dict[str, int]:
    flex_block_size = _validate_flex_block_size(flex_block_size)
    return {
        "BLOCK_M": gcd(flex_block_size, 128),
        "BLOCK_N": gcd(flex_block_size, 64),
    }


def _trie_parent_list(trie: TrieNode, padded_size: int) -> list[int]:
    parent = [-1] * padded_size
    for node in trie.nodes:
        parent_end_pos = node.ancestors[-1].end_idx if node.ancestors else -1
        if 0 <= node.start_idx < padded_size:
            parent[node.start_idx] = parent_end_pos
        for pos in range(node.start_idx + 1, node.end_idx + 1):
            if pos < padded_size:
                parent[pos] = pos - 1
    return parent


def _tree_euler_intervals(
    trie: TrieNode,
    padded_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU tin/tout arrays for O(1) ancestor checks."""
    num_tree_tokens = min(trie.num_tokens, padded_size)
    tin = torch.full((padded_size,), -1, dtype=torch.int32)
    tout = torch.full((padded_size,), -1, dtype=torch.int32)
    if num_tree_tokens == 0:
        return tin, tout

    parent = _trie_parent_list(trie, padded_size)
    children: list[list[int]] = [[] for _ in range(num_tree_tokens)]
    roots: list[int] = []
    for idx in range(num_tree_tokens):
        p = parent[idx]
        if 0 <= p < num_tree_tokens:
            children[p].append(idx)
        else:
            roots.append(idx)

    timer = 0

    def dfs(root: int) -> None:
        nonlocal timer
        stack: list[tuple[int, bool]] = [(root, False)]
        while stack:
            node, exiting = stack.pop()
            if exiting:
                tout[node] = timer
                continue
            tin[node] = timer
            timer += 1
            stack.append((node, True))
            for child in reversed(children[node]):
                stack.append((child, False))

    for root in roots:
        dfs(root)
    return tin, tout


def _build_tree_block_mask_from_trie(
    trie: TrieNode,
    padded_size: int,
    device: torch.device,
    flex_block_size: int,
    timings: dict[str, float] | None = None,
) -> BlockMask:
    """Build a PyTorch BlockMask from trie ancestry metadata."""
    block_mask_start = _get_time()
    flex_block_size = _validate_flex_block_size(flex_block_size)
    if padded_size % flex_block_size != 0:
        raise ValueError(
            f"padded_size={padded_size} must be a multiple of flex_block_size={flex_block_size}."
        )

    euler_start = _get_time()
    tin_cpu, tout_cpu = _tree_euler_intervals(trie, padded_size)
    euler_time = _get_time() - euler_start
    if timings is not None:
        timings["tree_euler_intervals_time"] = (
            timings.get("tree_euler_intervals_time", 0.0) + euler_time
        )
    num_tree_tokens = min(trie.num_tokens, padded_size)
    tin = tin_cpu.to(device)
    tout = tout_cpu.to(device)
    num_tree_tokens_tensor = torch.tensor(num_tree_tokens, dtype=torch.long, device=device)

    def tree_mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        q_valid = q_idx < num_tree_tokens_tensor
        kv_valid = kv_idx < num_tree_tokens_tensor
        q_tin = tin[q_idx]
        kv_tin = tin[kv_idx]
        kv_tout = tout[kv_idx]
        return q_valid & kv_valid & (kv_tin <= q_tin) & (q_tin < kv_tout)

    block_mask = create_block_mask(
        tree_mask_mod,
        B=1,
        H=1,
        Q_LEN=padded_size,
        KV_LEN=padded_size,
        device=device,
        BLOCK_SIZE=flex_block_size,
        _compile=False,
    )
    if timings is not None:
        timings["block_mask_time"] = (
            timings.get("block_mask_time", 0.0) + (_get_time() - block_mask_start)
        )
    return block_mask


def _tree_attention_meta_from_trie(
    trie: TrieNode,
    padded_size: int,
    device: torch.device,
    flex_block_size: int,
    timings: dict[str, float] | None = None,
) -> TreeAttentionMeta:
    meta_start = _get_time()
    meta = TreeAttentionMeta(
        block_mask=_build_tree_block_mask_from_trie(
            trie,
            padded_size,
            device,
            flex_block_size,
            timings=timings,
        )
    )
    if timings is not None:
        timings["tree_attn_meta_time"] = (
            timings.get("tree_attn_meta_time", 0.0) + (_get_time() - meta_start)
        )
    return meta


def _patch_tree_attention_for_compiled_flex(flex_block_size: int) -> None:
    """Use compiled PyTorch flex_attention for Archon tree attention locally.

    AReaL's Archon wrapper calls bare flex_attention, which falls back to an
    unfused math implementation. This process-local monkey patch keeps the
    AReaL checkout untouched while forcing the Archon tree path through
    torch.compile(flex_attention). The block mask still uses AReaL's default
    sparse block size controlled by DynamicTreeAttn.
    """
    global _PATCHED_TREE_ATTENTION, _PATCHED_TREE_ATTENTION_OPTIONS
    kernel_options = _flex_kernel_options(flex_block_size)
    if _PATCHED_TREE_ATTENTION:
        if _PATCHED_TREE_ATTENTION_OPTIONS != kernel_options:
            raise RuntimeError(
                "TreeAttentionWrapper is already patched with kernel options "
                f"{_PATCHED_TREE_ATTENTION_OPTIONS}, cannot repatch with {kernel_options}."
            )
        return

    compiled_flex_attention = torch.compile(
        flex_attention,
        dynamic=True,
        options={
            "epilogue_fusion": True,
            "max_autotune": False,
            "shape_padding": True,
            "trace.enabled": False,
            "triton.cudagraphs": False,
        },
    )

    def forward(
        self: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        tree_attn_meta: TreeAttentionMeta | None = None,
    ) -> torch.Tensor:
        batch = q.shape[0]
        assert batch == 1, (
            f"TreeAttentionWrapper expects batch=1 for packed sequences, got batch={batch}"
        )
        assert tree_attn_meta is not None, (
            "TreeAttentionWrapper requires tree_attn_meta. "
            "For standard attention, use VarlenAttentionWrapper instead."
        )
        if tree_attn_meta.triton_data is not None:
            raise RuntimeError(
                "Expected flex-attention TreeAttentionMeta, but got Triton metadata. "
                "Unset AREAL_USE_TRITON_TREE_ATTN to use compiled flex attention."
            )
        assert tree_attn_meta.block_mask is not None
        return compiled_flex_attention(
            q,
            k,
            v,
            block_mask=tree_attn_meta.block_mask,
            score_mod=None,
            scale=scale,
            enable_gqa=q.shape[1] != k.shape[1],
            kernel_options=kernel_options,
        )

    TreeAttentionWrapper.forward = forward
    _PATCHED_TREE_ATTENTION = True
    _PATCHED_TREE_ATTENTION_OPTIONS = kernel_options


def _patch_qwen3_sparse_ac_skip_attention() -> None:
    """Keep Archon AC off tree attention so checkpoint recompute cannot hit flex fallback."""
    global _PATCHED_QWEN3_SPARSE_AC
    if _PATCHED_QWEN3_SPARSE_AC:
        return

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )
    from areal.experimental.models.archon.qwen3.infra import parallelize as qwen3_parallelize

    def apply_ac_ffn_only(model: nn.Module, ac_config, **kwargs) -> None:
        if ac_config.mode == "none":
            return
        if ac_config.mode not in ("full", "selective"):
            raise ValueError(f"Unsupported sparse AC mode: {ac_config.mode}")
        if not hasattr(model, "layers"):
            raise ValueError("Model must have a 'layers' attribute to apply AC")

        ckpt_kwargs = dict(
            preserve_rng_state=ac_config.preserve_rng_state,
            determinism_check=ac_config.determinism_check,
            early_stop=ac_config.early_stop,
            debug=ac_config.debug,
        )
        for _, block in model.layers.named_children():
            if getattr(block, "feed_forward", None) is not None:
                block.feed_forward = checkpoint_wrapper(block.feed_forward, **ckpt_kwargs)
            if getattr(block, "moe", None) is not None:
                block.moe = checkpoint_wrapper(block.moe, **ckpt_kwargs)

    qwen3_parallelize.apply_ac = apply_ac_ffn_only
    _PATCHED_QWEN3_SPARSE_AC = True


@dataclass
class DenseMB:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    seq_indices: list[int]
    lengths: list[int]


@dataclass
class SparseMB:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    trie: TrieNode
    seq_indices: list[int]
    num_tree_tokens: int
    padded_size: int


class _BuildNode:
    __slots__ = ("token_id", "node_id", "children", "is_end", "sequence_ids")

    def __init__(self, token_id: int, node_id: int):
        self.token_id = token_id
        self.node_id = node_id
        self.children: dict[int, _BuildNode] = {}
        self.is_end = False
        self.sequence_ids: list[int] = []


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_time() -> float:
    _sync()
    return time.perf_counter()


def initialize_archon_engine(
    model_path: str,
    *,
    dtype: str,
    sparse: bool,
    max_tokens_per_mb: int,
    flex_block_size: int = DEFAULT_FLEX_BLOCK_SIZE,
    act_ckpt: bool = False,
    master_port: str = "29500",
) -> ArchonEngine:
    """Create a single-process-compatible Archon engine for direct model calls."""
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", master_port)

    if sparse:
        _patch_tree_attention_for_compiled_flex(flex_block_size)
        if act_ckpt:
            _patch_qwen3_sparse_ac_skip_attention()

    mb_spec = MicroBatchSpec(
        n_mbs=1,
        max_tokens_per_mb=max_tokens_per_mb if sparse else None,
    )
    config = TrainEngineConfig(
        backend="archon:d1",
        experiment_name="dynamic_tree_attn_bench",
        trial_name="archon",
        path=model_path,
        dtype=dtype,
        mb_spec=mb_spec,
        pad_to_maximum=sparse,
        gradient_checkpointing=act_ckpt,
        optimizer=None,
        enable_tree_training=sparse,
        archon=ArchonEngineConfig(attn_type="varlen"),
    )
    engine = ArchonEngine(config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    engine.create_process_group(
        parallel_strategy=ParallelStrategy(data_parallel_size=world_size)
    )
    engine.initialize(
        addr=None,
        ft_spec=FinetuneSpec(
            total_train_epochs=1,
            dataset_size=1,
            train_batch_size=1,
        ),
    )
    return engine


def zero_grad(engine: ArchonEngine) -> None:
    """Clear gradients without requiring an optimizer on the benchmark engine."""
    for model in engine.model_parts:
        model.zero_grad(set_to_none=True)


def _archon_model_forward(
    engine: ArchonEngine,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    tree_attn_meta: TreeAttentionMeta | None = None,
) -> torch.Tensor:
    logits = engine.model(
        input_ids,
        position_ids,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        tree_attn_meta=tree_attn_meta,
    )
    return logits.squeeze(0)


def _tree_token_count(token_seqs: list[torch.LongTensor], indices: list[int]) -> int:
    if not indices:
        return 0
    trie = TokenTrie([token_seqs[i] for i in indices])
    return int(trie.get_stats(mode="forward")["n_tree_tokens"])


def _original_tree_token_count(token_seqs: list[torch.LongTensor]) -> int:
    if not token_seqs:
        return 0
    trie = TokenTrie(token_seqs)
    return int(trie.get_stats(mode="forward")["n_tree_tokens"])


def _truncate_token_seqs(
    token_seqs: list[torch.LongTensor],
    max_tokens_per_mb: int,
) -> list[torch.LongTensor]:
    if max_tokens_per_mb <= 0:
        return token_seqs
    return [seq[:max_tokens_per_mb] for seq in token_seqs]


def _truncate_info(
    original_token_seqs: list[torch.LongTensor],
    truncated_token_seqs: list[torch.LongTensor],
    max_tokens_per_mb: int,
) -> dict[str, bool | int | str]:
    original_tokens = sum(seq.numel() for seq in original_token_seqs)
    truncated_tokens = sum(seq.numel() for seq in truncated_token_seqs)
    n_truncated_sequences = sum(
        int(orig.numel() != trunc.numel())
        for orig, trunc in zip(original_token_seqs, truncated_token_seqs)
    )
    truncated = n_truncated_sequences > 0
    info: dict[str, bool | int | str] = {
        "truncated": truncated,
        "n_tokens_before_truncate": original_tokens,
        "n_tokens_after_truncate": truncated_tokens,
        "n_truncated_sequences": n_truncated_sequences,
    }
    if truncated:
        info["warning"] = (
            f"truncated {n_truncated_sequences} sequences to max_tokens_per_mb="
            f"{max_tokens_per_mb}; tokens {original_tokens}->{truncated_tokens}"
        )
    return info


def _split_dense_bins(
    token_seqs: list[torch.LongTensor],
    max_tokens_per_mb: int,
) -> list[list[int]]:
    if max_tokens_per_mb <= 0:
        return [list(range(len(token_seqs)))]
    max_seq_len = max(seq.numel() for seq in token_seqs)
    if max_seq_len > max_tokens_per_mb:
        raise ValueError(
            f"Sequence length {max_seq_len} exceeds max_tokens_per_mb={max_tokens_per_mb}."
        )
    total_tokens = sum(seq.numel() for seq in token_seqs)
    k = max(1, ceil(total_tokens / max_tokens_per_mb))
    while k <= len(token_seqs):
        bins = [bucket for bucket in LB_by_n_tokens(token_seqs, k) if bucket]
        if all(sum(token_seqs[i].numel() for i in bucket) <= max_tokens_per_mb for bucket in bins):
            return bins
        k += 1
    return [[i] for i in range(len(token_seqs))]


def _split_sparse_bins(
    token_seqs: list[torch.LongTensor],
    max_tokens_per_mb: int,
    *,
    mode: str,
    block_size: int,
) -> list[list[int]]:
    if max_tokens_per_mb <= 0:
        return [list(range(len(token_seqs)))]
    max_seq_len = max(seq.numel() for seq in token_seqs)
    if max_seq_len > max_tokens_per_mb:
        raise ValueError(
            f"Sequence length {max_seq_len} exceeds max_tokens_per_mb={max_tokens_per_mb}."
        )

    time_model = TreeTimeModel()
    return [
        bucket
        for bucket in split_by_dfs_cost_limit(
            token_seqs,
            time_model,
            cost_limit=max_tokens_per_mb,
            mode=mode,
            block_size=block_size,
        )
        if bucket
    ]


def _insert_sequence(
    root: _BuildNode,
    all_nodes: list[_BuildNode],
    token_ids: torch.LongTensor,
    seq_id: int,
) -> None:
    current = root
    for token in token_ids.tolist():
        token = int(token)
        child = current.children.get(token)
        if child is None:
            child = _BuildNode(token, len(all_nodes))
            current.children[token] = child
            all_nodes.append(child)
        child.sequence_ids.append(seq_id)
        current = child
    current.is_end = True


def _compress_trie(root: _BuildNode) -> TrieNode:
    trie_root = TrieNode(tree_id=0)

    def compress_chain(node: _BuildNode, ancestors: list[TrieNode]) -> TrieNode:
        tokens: list[int] = []
        current = node
        start_id = node.node_id
        while True:
            tokens.append(current.token_id)
            if len(current.children) != 1 or current.is_end:
                break
            child = next(iter(current.children.values()))
            if current.sequence_ids != child.sequence_ids:
                raise ValueError("Sequence ids mismatch while compressing trie.")
            current = child

        trie_node = TrieNode(
            tree_id=0,
            start_idx=start_id,
            end_idx=current.node_id,
            tokens=tokens,
            sequence_ids=current.sequence_ids.copy(),
            ancestors=ancestors.copy(),
        )
        trie_root.nodes.append(trie_node)

        for token, child in sorted(current.children.items()):
            trie_node.children[token] = compress_chain(child, ancestors + [trie_node])
        return trie_node

    for token, child in sorted(root.children.items()):
        trie_root.children[token] = compress_chain(child, [])
    return trie_root


def _build_areal_trie_from_indices(
    token_seqs: list[torch.LongTensor],
    seq_indices: list[int],
) -> TrieNode:
    root = _BuildNode(-1, -1)
    all_nodes: list[_BuildNode] = []
    # Sort within the microbatch to match DynamicTreeAttn's trie-centric packing
    # while preserving original sequence ids in TrieNode.sequence_ids.
    for seq_id in sorted(seq_indices, key=lambda i: token_seqs[i].tolist()):
        _insert_sequence(root, all_nodes, token_seqs[seq_id], seq_id)
    return _compress_trie(root)


def _pack_sparse_input_ids(
    trie: TrieNode,
    *,
    padded_size: int,
    device: torch.device,
) -> torch.Tensor:
    input_ids = torch.zeros((padded_size,), dtype=torch.long, device=device)
    for node in trie.nodes:
        input_ids[node.start_idx : node.end_idx + 1] = torch.tensor(
            node.tokens,
            dtype=torch.long,
            device=device,
        )
    return input_ids.unsqueeze(0)


def _packed_tree_position_ids(
    trie: TrieNode,
    *,
    padded_size: int,
    device: torch.device,
) -> torch.Tensor:
    position_ids = torch.zeros((padded_size,), dtype=torch.long, device=device)
    for node in trie.nodes:
        depth = sum(ancestor.num_tokens for ancestor in node.ancestors)
        node_positions = torch.arange(
            depth,
            depth + node.num_tokens,
            dtype=torch.long,
            device=device,
        )
        position_ids[node.start_idx : node.end_idx + 1] = node_positions
    return position_ids.unsqueeze(0)


def _prepare_dense_mbs(
    token_seqs: list[torch.LongTensor],
    *,
    max_tokens_per_mb: int,
    device: torch.device,
) -> list[DenseMB]:
    mbs = []
    for seq_indices in _split_dense_bins(token_seqs, max_tokens_per_mb):
        lengths = [token_seqs[i].numel() for i in seq_indices]
        total_len = sum(lengths)
        input_ids = torch.empty((total_len,), dtype=torch.long, device=device)
        position_ids = torch.empty((total_len,), dtype=torch.long, device=device)
        cu_values = [0]
        cursor = 0
        for seq_id, length in zip(seq_indices, lengths):
            seq = token_seqs[seq_id].to(device)
            input_ids[cursor : cursor + length] = seq
            position_ids[cursor : cursor + length] = torch.arange(
                length, dtype=torch.long, device=device
            )
            cursor += length
            cu_values.append(cursor)
        cu_seqlens = torch.tensor(cu_values, dtype=torch.int32, device=device)
        mbs.append(
            DenseMB(
                input_ids=input_ids.unsqueeze(0),
                position_ids=position_ids.unsqueeze(0),
                cu_seqlens=cu_seqlens,
                max_seqlen=max(lengths),
                seq_indices=seq_indices,
                lengths=lengths,
            )
        )
    return mbs


def _prepare_sparse_mbs(
    token_seqs: list[torch.LongTensor],
    *,
    max_tokens_per_mb: int,
    mode: str,
    block_size: int,
    flex_block_size: int,
    device: torch.device,
) -> list[SparseMB]:
    flex_block_size = _validate_flex_block_size(flex_block_size)
    if max_tokens_per_mb % flex_block_size != 0:
        raise ValueError(
            f"max_tokens_per_mb={max_tokens_per_mb} must be a multiple of flex_block_size={flex_block_size}."
        )
    mbs = []
    for seq_indices in _split_sparse_bins(
        token_seqs,
        max_tokens_per_mb,
        mode=mode,
        block_size=block_size,
    ):
        trie = _build_areal_trie_from_indices(token_seqs, seq_indices)
        num_tree_tokens = trie.num_tokens
        input_ids = _pack_sparse_input_ids(
            trie,
            padded_size=max_tokens_per_mb,
            device=device,
        )
        position_ids = _packed_tree_position_ids(
            trie,
            padded_size=max_tokens_per_mb,
            device=device,
        )
        mbs.append(
            SparseMB(
                input_ids=input_ids,
                position_ids=position_ids,
                trie=trie,
                seq_indices=seq_indices,
                num_tree_tokens=num_tree_tokens,
                padded_size=max_tokens_per_mb,
            )
        )
    return mbs


def _dense_stats(
    *,
    loss: float,
    elapsed: float,
    token_seqs: list[torch.LongTensor],
) -> dict[str, float | int]:
    return {
        "loss": loss,
        "time": elapsed,
        "compute_time": elapsed,
        "n_sequences": len(token_seqs),
        "n_tokens": sum(seq.numel() for seq in token_seqs),
    }


@torch.no_grad()
def archon_dense_forward(
    engine: ArchonEngine,
    token_seqs: list[torch.LongTensor],
    *,
    max_tokens_per_mb: int,
) -> dict[str, float | int]:
    prepare_start = _get_time()
    original_token_seqs = token_seqs
    token_seqs = _truncate_token_seqs(token_seqs, max_tokens_per_mb)
    truncate_info = _truncate_info(original_token_seqs, token_seqs, max_tokens_per_mb)
    mbs = _prepare_dense_mbs(
        token_seqs,
        max_tokens_per_mb=max_tokens_per_mb,
        device=engine.device,
    )
    prepare_time = _get_time() - prepare_start

    start = _get_time()
    logprobs_list = []
    for mb in mbs:
        logits = _archon_model_forward(
            engine,
            mb.input_ids,
            mb.position_ids,
            cu_seqlens=mb.cu_seqlens,
            max_seqlen=mb.max_seqlen,
        )
        cursor = 0
        for length in mb.lengths:
            logprobs = gather_logprobs(
                logits[cursor : cursor + length - 1],
                mb.input_ids[0, cursor + 1 : cursor + length],
            )
            logprobs_list.append(logprobs)
            cursor += length
    loss = sum(-logprobs.mean().item() for logprobs in logprobs_list)
    compute_time = _get_time() - start
    stats = _dense_stats(loss=loss, elapsed=prepare_time + compute_time, token_seqs=token_seqs)
    stats["prepare_time"] = prepare_time
    stats["compute_time"] = compute_time
    stats["n_micro_batches"] = len(mbs)
    stats.update(truncate_info)
    return stats


def archon_dense_backward(
    engine: ArchonEngine,
    token_seqs: list[torch.LongTensor],
    attachs: list[dict],
    loss_fn: LossFn,
    *,
    max_tokens_per_mb: int,
) -> dict[str, float | int]:
    prepare_start = _get_time()
    original_token_seqs = token_seqs
    token_seqs = _truncate_token_seqs(token_seqs, max_tokens_per_mb)
    truncate_info = _truncate_info(original_token_seqs, token_seqs, max_tokens_per_mb)
    mbs = _prepare_dense_mbs(
        token_seqs,
        max_tokens_per_mb=max_tokens_per_mb,
        device=engine.device,
    )
    prepare_time = _get_time() - prepare_start

    start = _get_time()
    total_loss = 0.0
    for mb in mbs:
        logits = _archon_model_forward(
            engine,
            mb.input_ids,
            mb.position_ids,
            cu_seqlens=mb.cu_seqlens,
            max_seqlen=mb.max_seqlen,
        )
        cursor = 0
        mb_loss = None
        for seq_id, length in zip(mb.seq_indices, mb.lengths):
            logprobs, entropy = gather_logprobs_entropy(
                logits[cursor : cursor + length - 1],
                mb.input_ids[0, cursor + 1 : cursor + length],
            )
            seq_loss = loss_fn(logprobs, entropy, attachs[seq_id])
            mb_loss = seq_loss if mb_loss is None else mb_loss + seq_loss
            cursor += length
        if mb_loss is not None:
            mb_loss.backward()
            total_loss += mb_loss.detach().item()
    compute_time = _get_time() - start
    stats = _dense_stats(
        loss=total_loss,
        elapsed=prepare_time + compute_time,
        token_seqs=token_seqs,
    )
    stats["prepare_time"] = prepare_time
    stats["compute_time"] = compute_time
    stats["n_micro_batches"] = len(mbs)
    stats.update(truncate_info)
    return stats


def _sparse_stats(
    *,
    loss: float,
    prepare_time: float,
    compute_time: float,
    token_seqs: list[torch.LongTensor],
    mbs: list[SparseMB],
) -> dict[str, float | int]:
    n_tokens = sum(seq.numel() for seq in token_seqs)
    n_tree_tokens_original = _original_tree_token_count(token_seqs)
    n_tree_tokens = sum(mb.num_tree_tokens for mb in mbs)
    n_padded_tokens = sum(mb.padded_size for mb in mbs)
    return {
        "loss": loss,
        "time": prepare_time + compute_time,
        "prepare_time": prepare_time,
        "compute_time": compute_time,
        "n_sequences": len(token_seqs),
        "n_tokens": n_tokens,
        "n_tree_tokens_original": n_tree_tokens_original,
        "n_tree_tokens": n_tree_tokens,
        "n_padded_tokens": n_padded_tokens,
        "n_micro_batches": len(mbs),
        "tree_compressed_ratio_original": n_tokens / n_tree_tokens_original,
        "tree_compressed_ratio_mb": n_tokens / n_tree_tokens,
        "padding_ratio": n_padded_tokens / n_tree_tokens,
    }


@torch.no_grad()
def archon_sparse_forward(
    engine: ArchonEngine,
    token_seqs: list[torch.LongTensor],
    *,
    max_tokens_per_mb: int,
    flex_block_size: int = DEFAULT_FLEX_BLOCK_SIZE,
) -> dict[str, float | int]:
    prepare_start = _get_time()
    original_token_seqs = token_seqs
    token_seqs = _truncate_token_seqs(token_seqs, max_tokens_per_mb)
    truncate_info = _truncate_info(original_token_seqs, token_seqs, max_tokens_per_mb)
    mbs = _prepare_sparse_mbs(
        token_seqs,
        max_tokens_per_mb=max_tokens_per_mb,
        mode="forward",
        block_size=0,
        flex_block_size=flex_block_size,
        device=engine.device,
    )
    prepare_time = _get_time() - prepare_start

    compute_start = _get_time()
    total_loss = 0.0
    timings: dict[str, float] = {}
    for mb in mbs:
        tree_attn_meta = _tree_attention_meta_from_trie(
            mb.trie,
            mb.padded_size,
            mb.input_ids.device,
            flex_block_size,
            timings=timings,
        )
        seq_len = mb.input_ids.shape[-1]
        cu_seqlens = torch.tensor(
            [0, seq_len], dtype=torch.int32, device=mb.input_ids.device
        )
        logits = _archon_model_forward(
            engine,
            mb.input_ids,
            mb.position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
            tree_attn_meta=tree_attn_meta,
        )
        logprobs_by_seq = _gather_packed_tree_logprobs(
            logits,
            mb.trie,
            mb.input_ids,
        )
        # AReaL emits one rolled logprob per token; DynamicTreeAttn scores only
        # next-token predictions and excludes the synthetic final transition.
        total_loss += sum(
            -logprobs[:-1].mean().item() for logprobs in logprobs_by_seq.values()
        )
    compute_time = _get_time() - compute_start
    stats = _sparse_stats(
        loss=total_loss,
        prepare_time=prepare_time,
        compute_time=compute_time,
        token_seqs=token_seqs,
        mbs=mbs,
    )
    stats.update(timings)
    stats.update(truncate_info)
    return stats


def archon_sparse_backward(
    engine: ArchonEngine,
    token_seqs: list[torch.LongTensor],
    attachs: list[dict],
    loss_fn: LossFn,
    *,
    max_tokens_per_mb: int,
    block_size: int,
    flex_block_size: int = DEFAULT_FLEX_BLOCK_SIZE,
) -> dict[str, float | int]:
    prepare_start = _get_time()
    original_token_seqs = token_seqs
    token_seqs = _truncate_token_seqs(token_seqs, max_tokens_per_mb)
    truncate_info = _truncate_info(original_token_seqs, token_seqs, max_tokens_per_mb)
    mbs = _prepare_sparse_mbs(
        token_seqs,
        max_tokens_per_mb=max_tokens_per_mb,
        mode="backward",
        block_size=max(block_size, 1),
        flex_block_size=flex_block_size,
        device=engine.device,
    )
    prepare_time = _get_time() - prepare_start

    compute_start = _get_time()
    total_loss = 0.0
    timings: dict[str, float] = {}
    for mb in mbs:
        tree_attn_meta = _tree_attention_meta_from_trie(
            mb.trie,
            mb.padded_size,
            mb.input_ids.device,
            flex_block_size,
            timings=timings,
        )
        seq_len = mb.input_ids.shape[-1]
        cu_seqlens = torch.tensor(
            [0, seq_len], dtype=torch.int32, device=mb.input_ids.device
        )
        logits = _archon_model_forward(
            engine,
            mb.input_ids,
            mb.position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
            tree_attn_meta=tree_attn_meta,
        )
        logprobs_by_seq, entropy_by_seq = _gather_packed_tree_logprobs_entropy(
            logits,
            mb.trie,
            mb.input_ids,
        )
        mb_loss = None
        for seq_id in mb.trie.all_sequence_ids:
            seq_loss = loss_fn(
                logprobs_by_seq[seq_id][:-1],
                entropy_by_seq[seq_id][:-1],
                attachs[seq_id],
            )
            mb_loss = seq_loss if mb_loss is None else mb_loss + seq_loss
        if mb_loss is not None:
            mb_loss.backward()
            total_loss += mb_loss.detach().item()
    compute_time = _get_time() - compute_start
    stats = _sparse_stats(
        loss=total_loss,
        prepare_time=prepare_time,
        compute_time=compute_time,
        token_seqs=token_seqs,
        mbs=mbs,
    )
    stats.update(timings)
    stats.update(truncate_info)
    return stats
