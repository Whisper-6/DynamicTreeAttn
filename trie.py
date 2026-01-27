from dataclasses import dataclass, field
from typing import List, Tuple
import bisect

from typing import Optional
import random

@dataclass
class CTNode:
    """压缩Trie树节点"""
    depth: int = 0              # 节点深度
    seq_id: int = -1            # 字符串编号，-1表示内部节点
    main_Ld: int = 0            # min(儿子的main_Ld)，叶节点=自己的深度
    child_ids: List[int] = field(default_factory=list)  # 子节点ID列表
    
    def __post_init__(self):
        if self.child_ids is None:
            self.child_ids = []

class CompressedTrie:
    """压缩Trie树类，用于规划遍历顺序"""
    
    def __init__(self, lens: List[int], lcp_lens: List[int]):
        """
        初始化压缩Trie树
        
        Args:
            lens: 每个字符串的长度，保证按字典序排序
            lcp_lens: 相邻字符串的LCP长度，len(lcp_lens) = len(lens) - 1
        """
        if len(lcp_lens) != len(lens) - 1:
            raise ValueError("lcp_lens的长度必须为len(lens)-1")
        
        self.lens = lens
        self.lcp_lens = lcp_lens
        self.n_seqs = len(lens)
        self.lca_depth = -1
        
        # 存储所有节点
        self.nodes: List[CTNode] = []
        # 栈结构：(node_id, depth, seq_id)
        self.stack: List[Tuple[int, int, int]] = []
        
        self._build()
        self._compute_main_Ld()
    
    def _new_node(self, depth: int, seq_id: int = -1) -> int:
        """创建新节点，返回节点ID"""
        self.nodes.append(CTNode(depth=depth, seq_id=seq_id))
        return len(self.nodes) - 1
    
    def _pop_stack(self, depth: int):
        """弹出栈中深度大于depth的节点，连接到父节点"""
        while self.stack and self.stack[-1][1] > depth:
            # 弹出子节点
            child_id, child_depth, child_seq_id = self.stack.pop()
            
            if self.stack:
                # 获取父节点
                parent_id, parent_depth, parent_seq_id = self.stack[-1]
                # 将子节点连接到父节点
                self.nodes[parent_id].child_ids.append(child_id)
    
    def _build(self):
        """构建压缩Trie树（参考C++实现逻辑）"""
        # 创建根节点
        root_id = self._new_node(depth=0, seq_id=-1)
        self.stack.append((root_id, 0, -1))
        
        for seq_id in range(self.n_seqs):
            len_i = self.lens[seq_id]
            lcp = self.lcp_lens[seq_id - 1] if seq_id > 0 else 0

            # 从后往前找第一个深度 <= lcp 的节点作为父节点
            parent_idx = -1
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i][1] <= lcp:
                    parent_idx = i
                    break
            
            # 检查是否需要插入LCP节点
            if self.stack[parent_idx][1] < lcp:
                # 在栈中插入LCP节点
                lcp_node_id = self._new_node(depth=lcp, seq_id=-1)
                self.stack.insert(parent_idx + 1, (lcp_node_id, lcp, -1))
                # 更新父节点索引（现在LCP节点是新的父节点）
                parent_idx += 1
            
            # 弹出深度大于lcp的节点
            self._pop_stack(lcp)
            
            if lcp < len_i:
                # 创建新的叶节点
                cur_node_id = self._new_node(depth=len_i, seq_id=seq_id)
                self.stack.append((cur_node_id, len_i, seq_id))
        
        # 构建完成后，弹出栈中所有节点
        self._pop_stack(0)
    
    def _compute_main_Ld_dfs(self, node_id: int) -> int:
        """递归计算main_Ld值"""
        node = self.nodes[node_id]
        
        # 如果是叶节点
        if node.seq_id != -1:
            node.main_Ld = node.depth
            return node.depth
        
        # 如果是内部节点但没有子节点（理论上不会发生）
        if not node.child_ids:
            node.main_Ld = node.depth
            return node.depth
        
        # 计算所有子节点的main_Ld最小值
        min_main_Ld = float('inf')
        for child_id in node.child_ids:
            child_main_Ld = self._compute_main_Ld_dfs(child_id)
            min_main_Ld = min(min_main_Ld, child_main_Ld)
        
        node.main_Ld = min_main_Ld
        return min_main_Ld
    
    def _compute_main_Ld(self):
        """计算所有节点的main_Ld值"""
        if self.nodes:
            self._compute_main_Ld_dfs(0)
    
    def _dfs_with_children_order(self, node_id: int, child_order_func, 
                                 result: List[int], visited: Optional[List[bool]] = None) -> None:
        """
        通用DFS遍历函数，根据指定的子节点顺序函数进行遍历
        
        Args:
            node_id: 当前节点ID
            child_order_func: 函数，接收node_id返回子节点ID列表的顺序
            result: 收集结果的列表
            visited: 访问标记（可选）
        """
        node = self.nodes[node_id]
        
        # 如果是叶节点，记录字符串编号
        if node.seq_id != -1:
            result.append(node.seq_id)
            return
        
        # 根据指定的顺序函数获取子节点遍历顺序
        child_ids = child_order_func(node_id)
        
        # 递归遍历子节点
        for child_id in child_ids:
            self._dfs_with_children_order(child_id, child_order_func, result, visited)
    
    def _get_children_by_main_Ld(self, node_id: int) -> List[int]:
        """按main_Ld值排序子节点"""
        node = self.nodes[node_id]
        return sorted(
            node.child_ids,
            key=lambda child_id: self.nodes[child_id].main_Ld
        )
    
    def _get_children_by_main_Ld_2(self, node_id: int) -> List[int]:
        """按main_Ld值排序子节点"""
        node = self.nodes[node_id]
        return sorted(
            node.child_ids,
            key=lambda child_id: (0 if self.nodes[child_id].child_ids else 1, -self.nodes[child_id].main_Ld)
        )
    
    def _get_children_random(self, node_id: int, seed: Optional[int] = None) -> List[int]:
        """随机打乱子节点顺序"""
        node = self.nodes[node_id]
        child_ids = node.child_ids.copy()
        
        if seed is not None:
            local_random = random.Random(seed)
            local_random.shuffle(child_ids)
        else:
            random.shuffle(child_ids)
        
        return child_ids
    
    def get_str_order_by_main_Ld(self) -> List[int]:
        """获取按main_Ld优先DFS遍历得到的字符串顺序"""
        result = []
        self._dfs_with_children_order(0, self._get_children_by_main_Ld, result)
        return result
    
    def get_str_order_by_main_Ld_2(self) -> List[int]:
        """获取按main_Ld优先DFS遍历得到的字符串顺序"""
        result = []
        self._dfs_with_children_order(0, self._get_children_by_main_Ld_2, result)
        return result
    
    def get_str_order_random(self, seed: Optional[int] = None) -> List[int]:
        """获取随机打乱边表后的DFS遍历顺序"""
        result = []
        self._dfs_with_children_order(0, lambda node_id: self._get_children_random(node_id, seed), result)
        return result

def test_compressed_trie():
    lens1 = [5, 4, 3, 2]
    lcp_lens1 = [3, 2, 1]
    
    trie1 = CompressedTrie(lens1, lcp_lens1)
    
    order1 = trie1.get_str_order_by_main_Ld()
    print(order1)

if __name__ == "__main__":
    test_compressed_trie()