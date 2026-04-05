"""Analyze attention compression ratio for tree-structured sequence batches.

Computes two metrics per data file:
  1. Token compression ratio  = total_flat_tokens / trie_unique_tokens
  2. Attention FLOPs ratio    = flat_attn_ops / tree_attn_ops

For causal self-attention:
  - Flat: each sequence of length L costs L*(L+1)/2 attention ops.
  - Tree: each trie node of length N with A ancestor tokens costs
          N*A + N*(N+1)/2 attention ops.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch


@dataclass
class TrieNode:
    tokens: list[int] = field(default_factory=list)
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    seq_count: int = 0  # how many sequences pass through this node


def build_trie(sequences: list[torch.Tensor]) -> TrieNode:
    """Build a compressed trie from a list of token-id tensors."""
    root = _RawNode()
    for seq in sequences:
        _insert(root, seq.tolist())
    return _compress(root)


class _RawNode:
    __slots__ = ("children", "count")

    def __init__(self):
        self.children: dict[int, _RawNode] = {}
        self.count: int = 0  # sequences passing through


def _insert(root: _RawNode, tokens: list[int]):
    cur = root
    for t in tokens:
        if t not in cur.children:
            cur.children[t] = _RawNode()
        cur = cur.children[t]
        cur.count += 1


def _compress(raw_root: _RawNode) -> TrieNode:
    """Compress linear chains into single TrieNodes."""
    root = TrieNode()
    for token, child in raw_root.children.items():
        root.children[token] = _compress_chain(child, token)
    return root


def _compress_chain(node: _RawNode, first_token: int) -> TrieNode:
    tokens = [first_token]
    cur = node
    while len(cur.children) == 1:
        next_tok, next_node = next(iter(cur.children.items()))
        if next_node.count != cur.count:
            break
        tokens.append(next_tok)
        cur = next_node

    trie = TrieNode(tokens=tokens, seq_count=cur.count)
    for tok, child in cur.children.items():
        trie.children[tok] = _compress_chain(child, tok)
    return trie


@dataclass
class CompressionStats:
    filename: str
    num_seqs: int
    flat_tokens: int
    trie_tokens: int
    flat_attn_ops: int
    tree_attn_ops: int

    @property
    def token_ratio(self) -> float:
        return self.flat_tokens / self.trie_tokens if self.trie_tokens else float("inf")

    @property
    def attn_ratio(self) -> float:
        return self.flat_attn_ops / self.tree_attn_ops if self.tree_attn_ops else float("inf")


def count_trie_tokens(node: TrieNode) -> int:
    total = len(node.tokens)
    for child in node.children.values():
        total += count_trie_tokens(child)
    return total


def compute_tree_attn_ops(node: TrieNode, ancestor_tokens: int = 0) -> int:
    """Compute total attention operations for tree layout.

    For a node of length N with A ancestor tokens, cost = N*A + N*(N+1)/2.
    """
    N = len(node.tokens)
    A = ancestor_tokens
    ops = N * A + N * (N + 1) // 2
    for child in node.children.values():
        ops += compute_tree_attn_ops(child, A + N)
    return ops


def analyze_file(filepath: str) -> CompressionStats:
    sequences = torch.load(filepath, map_location="cpu")
    assert isinstance(sequences, list)

    num_seqs = len(sequences)
    lengths = [len(s) for s in sequences]
    flat_tokens = sum(lengths)
    flat_attn_ops = sum(L * (L + 1) // 2 for L in lengths)

    trie_root = build_trie(sequences)
    trie_tokens = count_trie_tokens(trie_root)
    tree_attn_ops = compute_tree_attn_ops(trie_root)

    return CompressionStats(
        filename=os.path.basename(filepath),
        num_seqs=num_seqs,
        flat_tokens=flat_tokens,
        trie_tokens=trie_tokens,
        flat_attn_ops=flat_attn_ops,
        tree_attn_ops=tree_attn_ops,
    )


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pt_files = sorted(
        [f for f in os.listdir(data_dir) if f.startswith("call") and f.endswith(".pt")],
        key=lambda x: int(x.replace("call", "").replace(".pt", "")),
    )

    print(f"{'File':<14} {'#Seq':>5} {'Flat Tok':>10} {'Trie Tok':>10} "
          f"{'Tok Ratio':>10} {'Flat Attn':>14} {'Tree Attn':>14} {'Attn Ratio':>11}")
    print("-" * 100)

    all_stats: list[CompressionStats] = []
    for f in pt_files:
        stats = analyze_file(os.path.join(data_dir, f))
        all_stats.append(stats)
        print(
            f"{stats.filename:<14} {stats.num_seqs:>5} {stats.flat_tokens:>10,} {stats.trie_tokens:>10,} "
            f"{stats.token_ratio:>10.2f}x {stats.flat_attn_ops:>14,} {stats.tree_attn_ops:>14,} "
            f"{stats.attn_ratio:>10.2f}x"
        )

    print("-" * 100)
    total_flat_tok = sum(s.flat_tokens for s in all_stats)
    total_trie_tok = sum(s.trie_tokens for s in all_stats)
    total_flat_attn = sum(s.flat_attn_ops for s in all_stats)
    total_tree_attn = sum(s.tree_attn_ops for s in all_stats)
    total_seqs = sum(s.num_seqs for s in all_stats)

    print(
        f"{'TOTAL':<14} {total_seqs:>5} {total_flat_tok:>10,} {total_trie_tok:>10,} "
        f"{total_flat_tok / total_trie_tok:>10.2f}x {total_flat_attn:>14,} {total_tree_attn:>14,} "
        f"{total_flat_attn / total_tree_attn:>10.2f}x"
    )

    print(f"\n{'='*60}")
    print(f"Average token compression ratio:     {total_flat_tok / total_trie_tok:.2f}x")
    print(f"Average attention FLOPs compression:  {total_flat_attn / total_tree_attn:.2f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
