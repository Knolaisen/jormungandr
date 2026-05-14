from torch import Tensor
import torch

def _compact_to_left(
    features: Tensor,
    position_embedding: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    # Mamba is a causal SSM: padded tokens in mid-sequence corrupt the running
    # hidden state for every following valid token. Row-major flatten of a 2-D
    # mask interleaves PAD with valid tokens, so we compact valid tokens to the
    # left and right-pad. Mamba state corruption is then confined to the tail,
    # which the decoder's cross-attention mask discards.
    sort_key = mask.to(torch.bool).to(torch.int32)
    order = torch.argsort(sort_key, dim=1, descending=True, stable=True)
    valid_counts = sort_key.sum(dim=1)
    new_sequence_length = int(valid_counts.max().item())
    order = order[:, :new_sequence_length]
    gather_index = order.unsqueeze(-1).expand(-1, -1, features.size(-1))
    compacted_features = torch.gather(features, dim=1, index=gather_index)
    compacted_position_embedding = torch.gather(
        position_embedding, dim=1, index=gather_index
    )
    arange = torch.arange(new_sequence_length, device=mask.device).unsqueeze(0)
    compacted_mask = (arange < valid_counts.unsqueeze(1)).to(mask.dtype)
    return compacted_features, compacted_position_embedding, compacted_mask
