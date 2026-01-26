import torch
from typing import List, Union
import tqdm
import time

from vocab_parallel import gather_logprobs_entropy

def backward(model, token_seqs: List[torch.LongTensor], attachs, loss_fn, gradient_checkpointing_enabled: bool) -> float:
    total_loss = 0.0
    model.train()

    if gradient_checkpointing_enabled:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    for i in tqdm.tqdm(range(len(token_seqs))):
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