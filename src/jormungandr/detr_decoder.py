from torch import nn, Tensor


class DETRDecoder(nn.Module):
    def __init__(self, num_queries: int = 16, hidden_dim: int = 16):
        super(DETRDecoder, self).__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # Additional layers can be added here
