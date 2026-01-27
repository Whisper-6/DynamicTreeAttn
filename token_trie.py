import torch
from typing import List, Union

from trie import CompressedTrie

def _lcp_torch(a: torch.Tensor, b: torch.Tensor) -> int:
    """Compute the length of the longest common prefix of two 1D tensors."""
    L = min(a.numel(), b.numel())
    eq = a[:L] == b[:L]
    return L if eq.all() else int((~eq).to(torch.int32).argmax().item())

def _calc_cost(lens, lcp_lens, bound) -> float:
    cost = 0
    for i in range(len(lens)):
        lcp = lcp_lens[i-1] if i > 0 else 0
        push_len = lens[i] - lcp
        cost += max(push_len, bound)

    return cost


def _leafization(input_ids: List[torch.LongTensor], attachs: List[dict]):
    """
        参数：
            input_ids: List of Token Tensor（按字典序排序）
            attachs: List of dict，表示每个 Token Tensor 的 loss 配置

        将完全重叠的前缀合并，并计算 lcp_lens 列表。
    """

    # 计算相邻序列的 LCP 长度，同时检查字典序
    lcp_lens = []
    for i in range(len(input_ids)-1):
        seq_L, seq_R = input_ids[i], input_ids[i+1]
        lcp = _lcp_torch(seq_L, seq_R)
        L = min(seq_L.numel(), seq_R.numel())
        if lcp < L and seq_L[lcp] > seq_R[lcp]:
            raise ValueError("Input_ids not sorted in lexicographic order.")
        lcp_lens.append(lcp)

    # 合并完全重叠的前缀，计算时只保留最长序列
    input_ids_leafed = []
    attach_lists = []
    lcp_lens_leafed = []

    fork = -1
    for i in range(len(input_ids)):
        if i == len(input_ids)-1 or lcp_lens[i] < min(input_ids[i].numel(), input_ids[i+1].numel()):
            input_ids_leafed.append(input_ids[i])
            if i < len(input_ids)-1:
                lcp_lens_leafed.append(lcp_lens[i])
            attach_list = []
            for k in range(fork+1, i+1):
                attach_list.append((attachs[k], input_ids[k].numel()))
            attach_lists.append(attach_list)
            fork = i

    return input_ids_leafed, attach_lists, lcp_lens_leafed

class TokenTrie:
    def __init__(
        self,
        inputs: List[torch.LongTensor],
        attachs: List[dict] | None = None,
        sorted: bool = False,
        dtype: torch.dtype = None,
    ):
        if attachs is not None:
            assert len(inputs) == len(attachs), "Length of inputs and attachs must match."
        else:
            attachs = [{} for _ in range(len(inputs))]
        
        # 向 attachs 中添加序列编号
        for seq_id in range(len(inputs)):
            attachs[seq_id]['_sequence_batch_id'] = seq_id

        # -------- sort by lexicographical order of input_ids --------
        if not sorted:
            pairs = list(zip(inputs, attachs))
            pairs.sort(key=lambda x: x[0].tolist())
            inputs_sorted, attachs_sorted = [p[0] for p in pairs], [p[1] for p in pairs]
        else:
            inputs_sorted, attachs_sorted = inputs, attachs
            
        # -------- leafization --------
        self.inputs, self.attach_lists, self.lcp_lens = \
            _leafization(inputs_sorted, attachs_sorted)

        self.lens = [len(ids) for ids in self.inputs]

        # -------- statistics --------
        self.n_sequences = len(inputs)
        self.n_tokens = sum(len(ids) for ids in inputs)
        self.n_leafed_tokens = sum(len(ids) for ids in self.inputs)
        self.n_tree_tokens = self.n_leafed_tokens - sum(self.lcp_lens)


    def count_task(self):
        if not self.lcp_lens:
            return 1
        min_lcp = min(self.lcp_lens)
        return self.lens.count(min_lcp) + 1
    
    def max_len(self):
        return max(self.lens)

    def get_forward_permute(self):
        trie = CompressedTrie(self.lens, self.lcp_lens)
        permutation = trie.get_str_order_by_main_Ld()
        return permutation

    def get_backward_permute(self):
        trie = CompressedTrie(self.lens, self.lcp_lens)
        permutation = trie.get_str_order_by_main_Ld_2()
        return permutation

    def get_random_permute(self):
        trie = CompressedTrie(self.lens, self.lcp_lens)
        permutation = trie.get_str_order_random()
        return permutation
    
    def permute(self, order):
        self.inputs = [self.inputs[i] for i in order]
        self.attach_lists = [self.attach_lists[i] for i in order]
        self.lcp_lens = [_lcp_torch(self.inputs[i], self.inputs[i+1]) for i in range(len(self.inputs)-1)]

    def forward_permute(self):
        order = self.get_forward_permute()
        self.permute(order)

    def backward_permute(self):
        order = self.get_backward_permute()
        self.permute(order)

    def random_permute(self):
        order = self.get_random_permute()
        self.permute(order)

    def try_devide(self, cost_limit: int, mem_bound: int) -> List[List[int]] | None:
        """
        Try to divide the sequences such that the maximum cost of each 
        part does not exceed cost_limit.
        
        If successful, return the division result (list of original 
        sequence IDs for each part); otherwise return None.
        """
        divs = [-1]

        start = 0
        for i in range(1, len(self.lens)):
            if _calc_cost(self.lens[start:i], self.lcp_lens[start:i], bound=mem_bound) > cost_limit:
                divs.append(i)
                start = i

        divs.append(len(self.lens))
        
        parts = []
        for i in range(len(divs)-1):
            part = []
            for j in range(divs[i]+1, divs[i+1]):
                for attachment, _ in self.attach_lists[j]:
                    part.append(attachment['_sequence_batch_id'])
            parts.append(part)
        
        return parts
                
    def divide(self, n_parts: int, mem_bound: int):
        """
        Divide the sequences into n_parts such that the maximum tree tokens
        in each part is minimized.
        """
        self.lens = [len(ids) for ids in self.inputs]

        L = 0
        R = _calc_cost(self.lens, self.lcp_lens, bound=mem_bound)

        while L < R:
            mid = (L + R) // 2
            parts = self.try_devide(cost_limit=mid, mem_bound=mem_bound)
            if parts is not None and len(parts) <= n_parts:
                R = mid
            else:
                L = mid + 1

        parts = self.try_devide(cost_limit=R, mem_bound=mem_bound)
        return parts