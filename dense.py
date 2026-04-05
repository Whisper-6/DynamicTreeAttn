import torch
from typing import List, Union
import tqdm

from vocab_parallel import gather_logprobs, gather_logprobs_entropy
from areal.utils.datapack import ffd_allocate

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


def backward_packed(
    model,
    token_seqs: List[torch.LongTensor],
    attachs,
    loss_fn,
    act_ckpt: bool,
    use_tqdm,
    mb_tokens: int,
) -> float:
    """Backward with FFD micro-batching and packed varlen input.

    - mb_tokens == -1 should be handled by caller and routed to ``backward``.
    - mb_tokens > -1 packs multiple sequences into a single [1, T_total] input
      and passes cu_seqlens-style metadata to the model.
    """
    if mb_tokens <= 0:
        raise ValueError(f"mb_tokens must be > 0 for packed mode, got {mb_tokens}.")

    total_loss = 0.0

    if act_ckpt:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    lengths = [int(seq.numel()) for seq in token_seqs]
    # Keep at least one sequence per micro-batch even if all fit into one bucket.
    group_indices = ffd_allocate(lengths, capacity=mb_tokens, min_groups=1)

    iterator = tqdm.tqdm(group_indices) if use_tqdm else group_indices
    for group in iterator:
        seqs = [token_seqs[i] for i in group]
        group_attachs = [attachs[i] for i in group]
        seq_lens = [int(seq.numel()) for seq in seqs]

        packed_ids = torch.cat(seqs, dim=0)
        packed_input_ids = packed_ids.unsqueeze(0).to(model.device)

        cu_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=model.device)
        cu_seqlens[1:] = torch.cumsum(
            torch.tensor(seq_lens, dtype=torch.int32, device=model.device), dim=0
        )
        max_seqlen = max(seq_lens)

        packed_loss = torch.tensor(0.0, device=model.device, dtype=torch.float32)
        try:
            outputs = model(
                input_ids=packed_input_ids,
                use_cache=False,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=int(max_seqlen),
                max_length_k=int(max_seqlen),
                attention_mask=dict(full_attention=None, sliding_attention=None),
            )
        except TypeError as exc:
            raise RuntimeError(
                "Model forward does not accept packed-sequence kwargs "
                "(cu_seq_lens_q/k, max_length_q/k). "
                "Please use --mb-tokens -1 or a packed-attention-capable model path."
            ) from exc
        logits = outputs.logits.squeeze(0)

        for i, attachment in enumerate(group_attachs):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            # seq_len <= 1 has no next-token prediction target
            if end - start <= 1:
                continue
            seq_logits = logits[start : end - 1]
            seq_labels = packed_ids[start + 1 : end].to(model.device)
            logprobs, entropy = gather_logprobs_entropy(logits=seq_logits, labels=seq_labels)
            loss = loss_fn(logprobs, entropy, attachment)
            packed_loss = packed_loss + loss

        packed_loss.backward()
        total_loss += packed_loss.item()

    return total_loss