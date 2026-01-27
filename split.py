import torch
from token_trie import TokenTrie

path = "/data/tree/DynamicTreeAttn/data/call3.pt"
intput_ids = torch.load(path)
token_trie = TokenTrie(intput_ids)
print(token_trie.count_task())
print(token_trie.max_len())