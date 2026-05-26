# QA

## TokenTrie 和 CompressedTrie 怎么分工？

`TokenTrie` 负责把原始输入序列整理成训练引擎可以线性消费的 leaf 序列。

它默认会先按 token 字典序排序，然后做 leafization：如果某条原始序列是另一条序列的前缀，这条短序列不会作为单独的 `inputs` 保留，而是挂到更长 leaf 的 `attach_lists` 里。因此 `TokenTrie.inputs` 只保留 leafization 后的叶子序列；非叶子终止序列通过 `(attachment, length)` 保存在对应 leaf 的 `attach_lists` 中。

`TokenTrie` 里主要有这些信息：

- `inputs`：当前 leaf 顺序下的 token 序列。
- `lens`：每个 leaf 序列的长度。
- `lcp_lens`：当前相邻 leaf 序列的 LCP 长度。
- `attach_lists`：每个 leaf 对应的原始序列 attachment 列表，里面的 `attachment["_sequence_batch_id"]` 是原始 batch id。

`CompressedTrie` 负责表示由 `lens/lcp_lens` 恢复出的压缩树结构，并提供不同遍历顺序的策略函数。

它的输入不是 token ids，而是：

```python
CompressedTrie(lens, lcp_lens)
```

它会显式构造压缩后的树节点。内部节点只记录 `depth` 和 `child_ids`，不保存 token 值；leaf 节点的 `seq_id` 是构造它时的 leaf 编号，也就是当时 `TokenTrie.inputs` 的下标，不是原始序列 id。

因此二者的关系是：

1. `TokenTrie` 决定当前有哪些 leaf，以及这些 leaf 当前按什么顺序排列。
2. `CompressedTrie` 在这个 leaf 顺序的基础上重建树结构。
3. `CompressedTrie.get_order_forward/backward()` 只返回一种 leaf 编号排列，表示推荐的遍历顺序。
4. `TokenTrie.forward_permute/backward_permute()` 会调用这些顺序函数，并把顺序落实回 `TokenTrie.inputs/lens/lcp_lens/attach_lists`。

也就是说，`CompressedTrie` 注重表示树的结构，并提供函数表示各种顺序；在给定的 `TokenTrie` 基础上，`TokenTrie` 会把这些顺序真正落实为自己的当前 leaf 顺序。

## leaf order 里存的是什么？

`CompressedTrie.get_order_forward/backward()` 返回的是 leaf 编号排列。

这个编号是 `CompressedTrie` 构建时的 `TokenTrie.inputs` 下标。例如构建时：

```text
TokenTrie.inputs[0] = A
TokenTrie.inputs[1] = B
TokenTrie.inputs[2] = C
```

那么 `get_order_backward()` 可能返回：

```text
[2, 0, 1]
```

含义是推荐按 `C, A, B` 的顺序遍历。这里的 `2/0/1` 不是原始 batch id。要回到原始序列 id，需要通过：

```python
token_trie.attach_lists[leaf_id][...][0]["_sequence_batch_id"]
```

如果先执行：

```python
token_trie.backward_permute()
```

那么 `TokenTrie` 会把这个顺序落实到自身，之后新的 `token_trie.inputs[0]` 就是旧的 `token_trie.inputs[2]`。

## 当前 flex attention 是怎么接入 tree attention 的？

当前本地实现不改 AReaL 仓库，而是在 `archon_bench.py` 里做两件事：

1. 自己从 `TrieNode` 构建 PyTorch `BlockMask`。
2. 在当前进程里 monkey patch `AReaL` 的 `TreeAttentionWrapper.forward`，让 Archon 的 tree attention 调用 `torch.compile(flex_attention)`。

执行链路是：

```text
run.py
  -> archon_bench.initialize_archon_engine(..., sparse=True)
      -> _patch_tree_attention_for_compiled_flex()
      -> ArchonEngine.initialize(...)

archon_sparse_forward/backward(...)
  -> _prepare_sparse_mbs(...)
  -> _tree_attention_meta_from_trie(...)
      -> _build_tree_block_mask_from_trie(...)
          -> BlockMask.from_kv_blocks(...)
  -> _archon_model_forward(..., tree_attn_meta=...)
  -> AReaL model attention layer
  -> patched TreeAttentionWrapper.forward(...)
  -> compiled_flex_attention(q, k, v, block_mask=..., score_mod=None, ...)
```

也就是说，`TreeAttentionMeta` 是每个 sparse microbatch 在 forward/backward 前构建的；真正的 `flex_attention` kernel 是模型 forward 走到 attention 层时触发的。

## tree mask 是怎么体现的？

tree attention 的语义是：query token 只能 attend 到它在 trie/tree 上的祖先 token。当前本地实现不用完整二维 attention matrix 来表达这个关系，而是把 trie 转成 Euler interval：

```text
kv 是 q 的祖先
<=> tin[kv] <= tin[q] < tout[kv]
```

`_build_tree_block_mask_from_trie()` 会先按 `BLOCK_SIZE` 粗粒度扫描 block pair：

- 如果一个 `(q_block, kv_block)` 里所有 token pair 都合法，放进 `full_kv_*`。
- 如果只有一部分 token pair 合法，放进 `kv_*` partial blocks。
- 如果完全没有合法 token pair，这个 block pair 不进入 `BlockMask`。

partial block 内部的精确 token-pair 判断由 `tree_mask_mod(batch, head, q_idx, kv_idx)` 完成。它不是读 dense mask，而是直接用上面的 Euler interval 判断：

```python
q_valid & kv_valid & (kv_tin <= q_tin) & (q_tin < kv_tout)
```

因此当前 patch 能保证 tree mask 语义，关键点是：

- block 级 metadata 决定哪些 block pair 会被 kernel 访问。
- partial block 内的 token 级合法性由 `mask_mod` 决定。
- full block 不再调用 `mask_mod`，所以只有当整个 block pair 全合法时才会被标成 full block。

## block_mask、mask_mod、score_mod 分别是什么？

`block_mask` 是传给 `flex_attention` 的整体稀疏结构对象。它里面包含：

- partial block 的数量和列索引。
- full block 的数量和列索引。
- `BLOCK_SIZE=(q_block_size, kv_block_size)`。
- `mask_mod`，也就是 partial block 里 token-pair 级别的 bool 判断函数。

`mask_mod(batch, head, q_idx, kv_idx) -> bool` 描述某个 query token 和某个 key/value token 是否允许 attention。当前 tree mask 就是在这里用 Euler interval 体现的。`batch/head` 是 batch 和 attention head 维度；当前 tree mask 与 batch/head 无关，所以函数里不使用它们。

`score_mod(score, batch, head, q_idx, kv_idx) -> score` 是修改 attention score 数值的函数。它不是用来描述 block 的，也不是当前 tree mask 的主体。当前本地 patch 传的是 `score_mod=None`，表示不额外改 score，只靠 `block_mask/mask_mod` 做遮罩。

## flex attention 的 block_size 现在是多少？

当前 flex attention 的稀疏 mask block size 是 `128 x 128`。

确认路径：

```text
AReaL/areal/models/tree_attn/constants.py
  BLOCK_SIZE = int(os.environ.get("AREAL_FLEX_ATTENTION_BLOCK_SIZE", "128"))

DynamicTreeAttn/archon_bench.py
  from areal.models.tree_attn.constants import BLOCK_SIZE
  BlockMask.from_kv_blocks(..., BLOCK_SIZE=(BLOCK_SIZE, BLOCK_SIZE), ...)
```

如果没有设置环境变量 `AREAL_FLEX_ATTENTION_BLOCK_SIZE`，默认就是 `128`。AReaL 初始化时也会打印类似：

```text
Using block mask in flex attention, block size: 128
```

注意这里的 `BLOCK_SIZE` 是 `BlockMask` 的稀疏 mask block 大小，不是 `run.py --block-size`。`--block-size` 在当前 benchmark 里用于 sparse backward 的 microbatch 切分和 cost model，不控制 flex attention kernel 的 block mask 粒度。

## BLOCK_M/BLOCK_N 和 block_size 是什么关系？

当前 compiled flex attention 调用里写了：

```python
kernel_options={
    "BLOCK_M": 128,
    "BLOCK_N": 64,
}
```

这里的 `BLOCK_M/BLOCK_N` 是 Inductor/Triton kernel 的计算 tile 大小：

- `BLOCK_M`：一次处理多少个 query 位置。
- `BLOCK_N`：一次处理多少个 key/value 位置。

它们不是 `BlockMask` 的 block size。当前配置下：

```text
BlockMask block size = 128 x 128
kernel tile          = 128 x 64
```

含义是：一个 mask block 在 query 方向覆盖一个 kernel tile，在 kv 方向覆盖两个 kernel tile。mask block 负责告诉 kernel 哪些区域可能有合法 attention；kernel tile 是实际计算时的分块方式。

之前把 `AREAL_FLEX_ATTENTION_BLOCK_SIZE` 改成 `64` 会报错，是因为 sparse mask 的 q block 变成了 64，但 Inductor 默认或当前 kernel 选择里的 `BLOCK_M` 仍然可能是 128。PyTorch flex attention lowering 要求 sparse mask block 能覆盖 kernel tile，不能出现一个 kernel tile 横跨多个 q mask block 的不兼容情况。所以 64 的 mask block 配 128 的 `BLOCK_M` 会触发 block size 约束错误。

## 为什么不走 AReaL 原来的 dense attention matrix？

AReaL 原始 flex 路径大致是：

```text
TreeAttentionMeta.from_trie(...)
  -> build_block_mask_from_trie(...)
      -> _build_attention_mask(trie, padded_size, device)
      -> create_block_mask_from_dense(attention_mask, ...)
```

`create_block_mask_from_dense()` 里定义的 `mask_mod` 是：

```python
return attention_mask[q_idx, k_idx]
```

所以 dense matrix 不只是“临时用来生成 block metadata”。`BlockMask` 会携带这个 `mask_mod`，partial block 里仍然会通过闭包读 `attention_mask[q_idx, k_idx]`。这就是本地 patch 要避开的地方。

当前本地实现直接构造 `BlockMask.from_kv_blocks(...)`，并让 `mask_mod` 使用 Euler interval 判断祖先关系，因此不需要保存或读取完整二维 attention matrix。

## 什么是 math fallback？

PyTorch 的 `flex_attention` 如果裸调用，没有通过 `torch.compile` lowering 到 fused/sparse kernel，就会走 unfused math 路径。这个路径会物化接近完整的 attention scores matrix，显存占用接近普通 dense attention。

之前看到的警告是这个意思：

```text
flex_attention called without torch.compile() - this will use an unfused implementation
that materializes the full scores matrix
```

当前本地 patch 用：

```python
compiled_flex_attention = torch.compile(flex_attention, ...)
```

并在 patched `TreeAttentionWrapper.forward` 里调用 compiled 版本，所以正常情况下不会走 math fallback。如果设置了 `AREAL_USE_TRITON_TREE_ATTN=1`，`TreeAttentionMeta` 会变成 Triton metadata；当前 patched flex forward 会直接报错，提示 unset 这个环境变量，因为用户要求的是 flex attention 路径。

## Archon sparse backward 的 activation checkpoint 应该用什么模式？

当前本仓库只通过 `archon_bench.py` 做本地 monkey patch，不修改 AReaL 源码。`run.py --run archon_sparse_backward --act-ckpt True` 会把 `act_ckpt` 传给 `initialize_archon_engine()`；当 `sparse=True and act_ckpt=True` 时，当前默认设置是：

```python
archon_config.ac_mode = os.environ.get("DTA_ARCHON_AC_MODE", "full")
```

因此默认是 `full`，不是 AReaL `ArchonEngineConfig` 的默认 `selective/op`。

AReaL 这几个 AC 配置含义如下：

- `ac_mode="full"`：每个 transformer block 都包 `checkpoint_wrapper`，backward 会重算整层，包括 tree/flex attention。
- `ac_mode="selective", selective_ac_option="1"`：layer-level selective AC，每 1 层包一次，效果基本等价于 full。
- `ac_mode="selective", selective_ac_option="2"`：每 2 层包一次，只重算约一半层。
- `ac_mode="selective", selective_ac_option="op"`：op-level selective AC。AReaL 使用 `create_selective_checkpoint_contexts`，policy 里对部分 op 倾向保存，对其他 op 倾向重算；`op_sac_save_list` 里包含 `torch._higher_order_ops.flex_attention`，并且 `mm` 有额外的隔次保存/重算策略。

在 `tau2_data/call1.pt`、`max_tokens_per_mb=4096`、Qwen3-0.6B、RTX 3090 上的实测：

| AC 配置 | tree attention 重算 | peak memory | flex fallback warning |
| --- | --- | ---: | --- |
| `full` | 28/28 modules, each called twice | 7.18 GB | no |
| `selective + 1` | 28/28 modules, each called twice | 7.18 GB | no |
| `selective + 2` | 14/28 modules recomputed | 10.66 GB | no |
| `selective + op` | 28/28 modules, each called twice | 12.18 GB | yes |

这里的调用次数来自 `DTA_DEBUG_TREE_ATTN_CALLS=1` 的本地调试汇总。例如 `full` 和 `selective + 1` 会输出：

```text
DTA_TREE_ATTN_SUMMARY modules=28 min_calls=2 max_calls=2 recomputed_modules=28
```

`selective + op` 的 fallback warning 不是因为 `TreeAttentionWrapper` 没被 patch；warning-as-error 的栈显示调用仍然进入了 `archon_bench.py` 的 patched `TreeAttentionWrapper.forward`，并调用 `compiled_flex_attention(...)`。但在 op-level selective checkpoint context 下，`torch.compile(flex_attention)` 的 wrapper 退回执行了原始 Python `flex_attention`，PyTorch 因此报：

```text
flex_attention called without torch.compile()
```

更具体地说，`selective/op` 会给 checkpoint forward/recompute 安装 SAC dispatch/context，用来按 op policy 决定保存或重算。这个 context 会干扰当前 `torch.compile(flex_attention)` higher-order op/lowering 路径，使 compiled wrapper 进入未编译的 `flex_attention` 实现。`full` 和 layer-level selective (`"1"`, `"2"`) 不安装 op-level SAC context，所以不会触发这个 fallback。

因此当前建议：Archon sparse tree/flex attention 不使用 `selective/op`。默认保持 `full`；如果需要用 selective 语义，优先用 `selective_ac_option="1"` 或按层频率调节的数字选项。

## dense 和 archon_dense 的 microbatch 是怎么组织的？

当前 `dense_forward/backward` 是按序列逐条跑的，不做 packed microbatch。代码里对 `token_seqs` 做循环，每次把一条序列 `unsqueeze(0)` 成 `[1, L]` 后调用 HF model。`--max-tokens-per-mb` 在这条路径里只用于截断单条序列长度，不会把多条序列合成一个 batch。

`archon_dense_forward/backward` 会做 packed dense microbatch。它先按 `max_tokens_per_mb` 把多条序列分桶，然后把同一个 microbatch 里的 token 串接成 `[1, total_len]`，同时传入 `cu_seqlens` 和每条序列自己的 `position_ids`。forward 得到 logits 后，再按每条序列的 `length` 切回来分别算 logprob/loss。

所以对比时要注意：

- `dense`：分序列执行，batch 维度一直是 1。
- `archon_dense`：一个 microbatch 内多序列 packed，但不做 tree prefix 压缩。
- `archon_sparse`：才是 trie/tree 形式的 prefix sharing + sparse/tree attention。

## 这次 dense forward 的结果怎么解读？

这组记录使用修正后的 `dense.py` 重跑。修正点是：HF dense 调用 model forward 时不再传 `labels=input_ids`，避免 HF `Qwen3ForCausalLM.forward` 内部额外计算 causal LM loss。现在 HF dense 和 Archon dense 都统一成 model forward 只产 logits，logprob/loss 在 benchmark 外层计算。

重跑配置：

```text
model: /data/jiarui/dta/models/Qwen3-0.6B
data:  ./tau2_data/call1.pt
run:   forward
max_tokens_per_mb: 6016
block_size: 1024
GPU:   CUDA_VISIBLE_DEVICES=0, RTX 3090
```

结果如下：

| 路径 | attn 实现 | loss | time | compute time | peak memory |
| --- | --- | ---: | ---: | ---: | ---: |
| `dense_forward` | `sdpa` | 192.655735 | 38.84 s | 38.84 s | 6.12 GB |
| `dense_forward` | `flash_attention_2` | 192.699789 | 39.86 s | 39.86 s | 6.12 GB |
| `archon_dense_forward` | Archon dense packed | 192.738379 | 39.34 s | 39.31 s | 4.88 GB |

这次数据里，HF `dense_forward` 的 `sdpa` 和 `flash_attention_2` 时间仍然非常接近；当前这组数里 `sdpa` 更快约 2.6%，但这个差距不应过度解读，最好以多轮重复的均值为准。

`archon_dense_forward` 不再明显快于 HF dense：相比 `sdpa` 的 38.84 s，它慢约 1.3%；相比 HF `flash_attention_2` 的 39.86 s，它快约 1.3%。这个量级已经接近运行波动和实现细节差异。

显存方面，`archon_dense_forward` 仍然更低：4.88 GB vs HF dense 6.12 GB。旧记录里 HF dense 的 12.93 GB 峰值主要来自传入 `labels` 后触发的 HF internal loss；这个 path 会对 logits 做 fp32 cross entropy，和外层 `gather_logprobs` 重复。

这里的主要结论是：`tau2_data/call1.pt` 的序列本来都比较长，dense packing 对总 attention 计算量影响有限。之前“Archon dense 快约 8%”的结论不可靠，主要被 HF dense 重复计算 internal loss 污染了。
