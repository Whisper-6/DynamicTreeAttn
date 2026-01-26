import torch
from typing import List, Union
import tqdm

from vocab_parallel import gather_logprobs

def forward(model, token_seqs: List[torch.LongTensor]) -> float:
    total_loss = 0.0

    logprobs_list = []

    for i in tqdm.tqdm(range(len(token_seqs))):
        input_ids = token_seqs[i].unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
        
        logprobs = gather_logprobs(
            logits=outputs.logits,
            labels=input_ids[:, 1:]
        )
        logprobs = logprobs.squeeze(0)
        logprobs_list.append(logprobs)

    return logprobs_list