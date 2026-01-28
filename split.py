import torch
from token_trie import TokenTrie

path = "/data/tree/tree-data/gsm8k/call_1_1.pt"
intput_ids = torch.load(path)
print(len(intput_ids))
token_trie = TokenTrie(intput_ids)
print(token_trie.count_task())
print(token_trie.max_len())