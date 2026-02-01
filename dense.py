import torch
from typing import List, Union
import tqdm

from vocab_parallel import gather_logprobs, gather_logprobs_entropy

@torch.no_grad()
def forward(model, token_seqs: List[torch.LongTensor], use_tqdm) -> List[torch.Tensor]:
    logprobs_list = []

    iterator = tqdm.tqdm(range(len(token_seqs))) if use_tqdm else range(len(token_seqs))
    for i in iterator:
        input_ids = token_seqs[i].unsqueeze(0).to(model.device)

        outputs = model(input_ids=input_ids, labels=input_ids)
        
        logprobs = gather_logprobs(
            logits=outputs.logits,
            labels=input_ids[:, 1:]
        )
        logprobs = logprobs.squeeze(0)
        logprobs_list.append(logprobs)

    return logprobs_list

def backward(model, token_seqs: List[torch.LongTensor], attachs, loss_fn, act_ckpt: bool, use_tqdm) -> float:
    total_loss = 0.0

    if act_ckpt:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    iterator = tqdm.tqdm(range(len(token_seqs))) if use_tqdm else range(len(token_seqs))
    for i in iterator:
        input_ids = token_seqs[i].unsqueeze(0).to(model.device)
        attachment = attachs[i]

        outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)

        logprobs, entropy = gather_logprobs_entropy(
            logits=outputs.logits,
            labels=input_ids[:, 1:]
        )
        logprobs = logprobs.squeeze(0)
        entropy = entropy.squeeze(0)

        loss = loss_fn(logprobs, entropy, attachment)

        loss.backward()

        total_loss += loss.item()
    
    return total_loss