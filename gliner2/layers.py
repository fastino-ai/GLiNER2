import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mlp(
    input_dim,
    intermediate_dims,
    output_dim,
    dropout=0.1,
    activation="gelu",
    add_layer_norm=False,
):
    """
    Creates a multi-layer perceptron (MLP) with specified dimensions and activation functions.
    """
    activation_mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
    }
    layers = []
    in_dim = input_dim
    for dim in intermediate_dims:
        layers.append(nn.Linear(in_dim, dim))
        if add_layer_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(activation_mapping[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class DownscaledTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4, num_layers=2, dropout=0.1):
        """
        Initializes a downscaled transformer with specified parameters.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.in_projector = nn.Linear(input_size, hidden_size)

        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)

        self.out_projector = create_mlp(
            input_dim=hidden_size + input_size,
            intermediate_dims=[input_size, input_size],
            output_dim=input_size,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (L, M, input_size).
        Returns:
            Tensor: Output tensor of shape (L, M, input_size).
        """
        original_x = x
        # Project input to hidden size.
        x = self.in_projector(x)
        # Apply transformer encoder.xx
        x = self.transformer(x)
        # Concatenate original input with transformer output.
        x = torch.cat([x, original_x], dim=-1)
        # Project back to input size.
        x = self.out_projector(x)
        return x


class CountLSTM(nn.Module):
    def __init__(self, hidden_size, max_count=20):
        """
        Initializes the module with a learned positional embedding for count steps and a GRU.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        # Learned positional embeddings for count steps: shape (max_count, hidden_size)
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        # Use a GRU layer; input shape is (seq_len, batch, input_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        # Projector layer: combines GRU output with original embeddings.
        self.projector = create_mlp(
            input_dim=hidden_size * 2,
            intermediate_dims=[hidden_size * 4],
            output_dim=hidden_size,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )

    def forward(self, pc_emb: torch.Tensor, gold_count_val: int) -> torch.Tensor:
        """
        Args:
            pc_emb (Tensor): Field embeddings of shape (M, hidden_size).
            gold_count_val (int): Predicted count value (number of steps).
        Returns:
            Tensor: Count-aware structure embeddings of shape (gold_count_val, M, hidden_size).
        """
        M, D = pc_emb.shape
        # Cap gold_count_val by max_count.
        gold_count_val = min(gold_count_val, self.max_count)
        # Create a sequence of count indices: shape (gold_count_val,)
        count_indices = torch.arange(gold_count_val, device=pc_emb.device)
        # Get positional embeddings for each count: (gold_count_val, hidden_size)
        pos_seq = self.pos_embedding(count_indices)
        # Expand pos_seq over the batch dimension: (gold_count_val, M, hidden_size)
        pos_seq = pos_seq.unsqueeze(1).expand(gold_count_val, M, D)
        # Initialize the GRU hidden state with the field embeddings.
        h0 = pc_emb.unsqueeze(0)  # shape: (1, M, hidden_size)
        # Run the GRU over the count sequence.
        output, _ = self.gru(pos_seq, h0)
        # Concatenate the GRU outputs with the original field embeddings.
        return self.projector(
            torch.cat([output, pc_emb.unsqueeze(0).expand_as(output)], dim=-1)
        )


class CountLSTMv2(nn.Module):
    def __init__(self, hidden_size, max_count=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.transformer = DownscaledTransformer(
            hidden_size,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
        )

    # NOTE: gold_count_val is now a 0-D Tensor, not a Python int
    def forward(self, pc_emb: torch.Tensor, gold_count_val: int) -> torch.Tensor:
        M, D = pc_emb.size()  # symbolic sizes

        # clamp without dropping to Python
        gold_count_val = min(gold_count_val, self.max_count)

        # build the *full* index vector once, then slice – ONNX supports both ops
        full_idx = torch.arange(self.max_count, device=pc_emb.device)
        count_idx = full_idx[:gold_count_val]  # (gold_count_val,)

        pos_seq = self.pos_embedding(count_idx)  # (gold_count_val, D)
        pos_seq = pos_seq.unsqueeze(1).expand(-1, M, -1)  # (gold_count_val, M, D)

        h0 = pc_emb.unsqueeze(0)  # (1, M, D)
        output, _ = self.gru(pos_seq, h0)  # (gold_count_val, M, D)

        pc_broadcast = pc_emb.unsqueeze(0).expand_as(output)
        return self.transformer(output + pc_broadcast)


class CountLSTMoE(nn.Module):
    """
    Count-aware module with a Mixture-of-Experts projector.

    Args
    ----
    hidden_size : int
        Model dimensionality (D).
    max_count   : int
        Maximum # count steps L.
    n_experts   : int, optional
        Number of FFN experts (default = 4).
    ffn_mult    : int, optional
        Expansion factor for expert bottleneck (default = 2 → inner = 2 D).
    dropout     : float, optional
        Drop-out used inside expert FFNs.
    """

    def __init__(
        self,
        hidden_size: int,
        max_count: int = 20,
        n_experts: int = 4,
        ffn_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size, self.max_count, self.n_experts = (
            hidden_size,
            max_count,
            n_experts,
        )

        # ───── positional encoding + recurrent core ─────
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        # ───── expert parameters (all packed in two tensors) ─────
        inner = hidden_size * ffn_mult
        # W1 : [E, D, inner]    b1 : [E, inner]
        self.w1 = nn.Parameter(torch.empty(n_experts, hidden_size, inner))
        self.b1 = nn.Parameter(torch.zeros(n_experts, inner))
        # W2 : [E, inner, D]  b2 : [E, D]
        self.w2 = nn.Parameter(torch.empty(n_experts, inner, hidden_size))
        self.b2 = nn.Parameter(torch.zeros(n_experts, hidden_size))

        # better than default init for large mats
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

        self.dropout = nn.Dropout(dropout)

        # ───── router / gating network ─────
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_experts),  # logits
            nn.Softmax(dim=-1),  # gates sum-to-1
        )

    # ---------------------------------------------------
    def forward(self, pc_emb: torch.Tensor, gold_count_val: int) -> torch.Tensor:
        """
        pc_emb : [M, D]    field embeddings
        gold_count_val : int   (# count steps to unroll)
        returns : [L, M, D]    count-aware embeddings
        """
        M, D = pc_emb.shape
        L = min(gold_count_val, self.max_count)

        idx = torch.arange(L, device=pc_emb.device)
        pos_seq = self.pos_embedding(idx).unsqueeze(1).expand(L, M, D)

        h0 = pc_emb.unsqueeze(0)  # [1, M, D]
        h, _ = self.gru(pos_seq, h0)  # [L, M, D]

        # ───── routing / gating ─────
        gates = self.router(h)  # [L, M, E]

        # ───── expert FFN: run *all* experts in parallel ─────
        # 1st linear
        x = torch.einsum("lmd,edh->lmeh", h, self.w1) + self.b1  # [L, M, E, inner]
        x = F.gelu(x)
        x = self.dropout(x)
        # 2nd linear
        x = torch.einsum("lmeh,ehd->lmed", x, self.w2) + self.b2  # [L, M, E, D]

        # ───── mixture weighted by gates ─────
        out = (gates.unsqueeze(-1) * x).sum(dim=2)  # [L, M, D]
        return out
