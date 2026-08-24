import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CompileSafeGRU(nn.Module):
    """Drop-in for single-layer ``nn.GRU`` that torch.compile can trace.

    Uses the same parameter names (``weight_ih_l0``, ``weight_hh_l0``,
    ``bias_ih_l0``, ``bias_hh_l0``) so pretrained checkpoints that were
    saved with ``nn.GRU`` load without any key remapping.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Run GRU over a sequence.

        Args:
            x: (seq_len, batch, input_size) input sequence.
            h: (batch, hidden_size) initial hidden state.

        Returns:
            (seq_len, batch, hidden_size) — hidden state at each step.
        """
        seq_len = x.shape[0]
        if seq_len == 0:
            return x.new_empty(0, h.shape[0], self.hidden_size)

        outputs = []
        for t in range(seq_len):
            gi = F.linear(x[t], self.weight_ih_l0, self.bias_ih_l0)
            gh = F.linear(h, self.weight_hh_l0, self.bias_hh_l0)

            i_r, i_z, i_n = gi.chunk(3, dim=-1)
            h_r, h_z, h_n = gh.chunk(3, dim=-1)

            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)

            h = (1 - z) * n + z * h
            outputs.append(h)

        return torch.stack(outputs, dim=0)


def create_mlp(input_dim, intermediate_dims, output_dim, dropout=0.1, activation="gelu", add_layer_norm=False):
    """
    Creates a multi-layer perceptron (MLP) with specified dimensions and activation functions.
    """
    activation_mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU
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
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)

        self.out_projector = create_mlp(
            input_dim=hidden_size + input_size,
            intermediate_dims=[input_size, input_size],
            output_dim=input_size,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
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
        # Compile-safe GRU (same parameter names as nn.GRU for checkpoint compat)
        self.gru = CompileSafeGRU(input_size=hidden_size, hidden_size=hidden_size)
        # Projector layer: combines GRU output with original embeddings.
        self.projector = create_mlp(
            input_dim=hidden_size * 2,
            intermediate_dims=[hidden_size * 4],
            output_dim=hidden_size,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
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
        # Run compile-safe GRU.
        output = self.gru(pos_seq, pc_emb)
        # Concatenate the GRU outputs with the original field embeddings.
        return self.projector(torch.cat([output, pc_emb.unsqueeze(0).expand_as(output)], dim=-1))


class CountLSTMv2(nn.Module):
    def __init__(self, hidden_size, max_count=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = CompileSafeGRU(hidden_size, hidden_size)
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

        # Compile-safe GRU forward.
        output = self.gru(pos_seq, pc_emb)  # (gold_count_val, M, D)

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

    def __init__(self,
                 hidden_size: int,
                 max_count: int = 20,
                 n_experts: int = 4,
                 ffn_mult: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size, self.max_count, self.n_experts = (
            hidden_size, max_count, n_experts)

        # ───── positional encoding + recurrent core ─────
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = CompileSafeGRU(hidden_size, hidden_size)

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

        # Compile-safe GRU forward.
        h = self.gru(pos_seq, pc_emb)  # [L, M, D]

        # ───── routing / gating ─────
        gates = self.router(h)  # [L, M, E]

        # ───── expert FFN: run *all* experts in parallel ─────
        # 1st linear
        x = torch.einsum('lmd,edh->lmeh', h, self.w1) + self.b1  # [L, M, E, inner]
        x = F.gelu(x)
        x = self.dropout(x)
        # 2nd linear
        x = torch.einsum('lmeh,ehd->lmed', x, self.w2) + self.b2  # [L, M, E, D]

        # ───── mixture weighted by gates ─────
        out = (gates.unsqueeze(-1) * x).sum(dim=2)  # [L, M, D]
        return out


# =============================================================================
# Span Representation Layer (from gliner package - copied to remove dependency)
# =============================================================================

def create_projection_layer(hidden_size: int, dropout: float, out_dim: Optional[int] = None) -> nn.Sequential:
    """Creates a two-layer projection network with ReLU activation and dropout.

    The projection layer expands the input by 4x in the hidden layer before
    projecting to the output dimension.

    Args:
        hidden_size: Size of the input hidden dimension.
        dropout: Dropout probability applied after the first layer.
        out_dim: Output dimension size. If None, uses hidden_size. Defaults to None.

    Returns:
        A Sequential module containing the projection layers.
    """
    if out_dim is None:
        out_dim = hidden_size

    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(out_dim * 4, out_dim)
    )


def extract_elements(sequence, indices):
    """Extract elements from a sequence using provided indices.

    Args:
        sequence (torch.Tensor): Input sequence of shape [B, L, D].
        indices (torch.Tensor): Indices to extract, shape [B, K].

    Returns:
        torch.Tensor: Extracted elements of shape [B, K, D].
    """
    D = sequence.size(-1)

    # Expand indices to [B, K, D]
    expanded_indices = indices.unsqueeze(2).expand(-1, -1, D)

    # Gather the elements
    extracted_elements = torch.gather(sequence, 1, expanded_indices)

    return extracted_elements


class SpanMarkerV0(nn.Module):
    """Marks and projects span endpoints using an MLP.

    A cleaner version of SpanMarker using the create_projection_layer utility.

    Attributes:
        max_width (int): Maximum span width to represent.
        project_start (nn.Module): MLP for projecting start positions.
        project_end (nn.Module): MLP for projecting end positions.
        out_project (nn.Module): Final projection layer.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        """Initialize the SpanMarkerV0 layer.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
            dropout (float, optional): Dropout rate. Defaults to 0.4.
        """
        super().__init__()
        self.max_width = max_width
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)

        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        """Compute span representations using start and end markers.

        Args:
            h (torch.Tensor): Token representations of shape [B, L, D].
            span_idx (torch.Tensor): Span indices of shape [B, *, 2].

        Returns:
            torch.Tensor: Span representations of shape [B, L, max_width, D].
        """
        B, L, D = h.size()

        start_rep = self.project_start(h)
        end_rep = self.project_end(h)

        start_span_rep = extract_elements(start_rep, span_idx[:, :, 0])
        end_span_rep = extract_elements(end_rep, span_idx[:, :, 1])

        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()

        return self.out_project(cat).view(B, L, self.max_width, D)

    def score_queries(
        self,
        h: torch.Tensor,
        queries: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        *,
        mask_invalid: bool = True,
    ) -> torch.Tensor:
        """Score spans directly against query vectors without final span materialization.

        For valid spans this is algebraically equivalent to ``forward()``
        followed by a dot product with ``queries``.  The first layer of
        ``out_project`` is split
        into start/end halves and the final layer is folded into the queries.

        Dropout remains in the same locations as the legacy path, so this
        factorization is valid in training as well as inference.  Set
        ``mask_invalid=False`` when the caller applies the legacy span mask
        after computing a loss; using ``-inf`` logits before BCE can otherwise
        produce ``inf * 0 -> nan``.  Invalid unmasked entries are finite padding
        values and are intentionally ignored by that post-loss validity mask.

        Args:
            h: Token representations ``[B, L, H]``.
            queries: Query vectors ``[B, Q, H]``.
            text_lengths: Optional true sequence lengths ``[B]`` for padded input.
            mask_invalid: Fill invalid spans with ``-inf`` when True.

        Returns:
            Raw span logits ``[B, Q, L, max_width]``.
        """
        B, L, H = h.shape
        if queries.ndim != 3 or queries.shape[0] != B or queries.shape[-1] != H:
            raise ValueError(
                "queries must have shape [batch, num_queries, hidden_size]"
            )

        first_linear = self.out_project[0]   # 2H -> 4H
        dropout = self.out_project[2]        # same module/location as legacy
        final_linear = self.out_project[3]   # 4H -> H

        # Legacy computes ReLU(cat(start_rep, end_rep)).  ReLU is elementwise
        start_rep = self.project_start(h).relu()
        end_rep = self.project_end(h).relu()

        # Split W1 = [W_start | W_end].  These large projections are now done
        # once per token rather than once per candidate span.
        start_hidden = F.linear(start_rep, first_linear.weight[:, :H], None)
        end_hidden = F.linear(end_rep, first_linear.weight[:, H:], None)

        starts = torch.arange(L, device=h.device).unsqueeze(1)
        widths = torch.arange(self.max_width, device=h.device).unsqueeze(0)
        ends = starts + widths

        valid = (ends < L).unsqueeze(0).expand(B, -1, -1)
        if text_lengths is not None:
            valid = valid & (ends.unsqueeze(0) < text_lengths[:, None, None])

        # Consecutive span ends are sliding windows over end_hidden.  Padding by
        # max_width - 1 and unfolding creates a strided view instead of two
        # [B, L, W, 4H] advanced-indexing gathers.  The start contribution stays
        # [B, L, 1, 4H] and broadcasts across widths.
        end_padded = F.pad(
            end_hidden, (0, 0, 0, self.max_width - 1)
        )
        end_windows = end_padded.unfold(1, self.max_width, 1).permute(0, 1, 3, 2)
        hidden = start_hidden.unsqueeze(2) + end_windows
        if first_linear.bias is not None:
            hidden = hidden + first_linear.bias
        hidden = hidden.relu()

        # Preserve the legacy dropout placement and flattened element order.
        # Keeping this flattened also feeds baddbmm directly without reshaping
        # back to [B, L, W, 4H].
        hidden = dropout(hidden.reshape(B, L * self.max_width, -1))

        # q^T(W2 d + b2) = (q^T W2)d + q^T b2
        folded_queries = torch.matmul(queries, final_linear.weight)
        folded_queries_t = folded_queries.transpose(1, 2)
        if final_linear.bias is not None:
            query_bias = torch.matmul(queries, final_linear.bias)
            logits = torch.baddbmm(
                query_bias[:, None, :], hidden, folded_queries_t
            )
        else:
            logits = torch.bmm(hidden, folded_queries_t)
        logits = logits.transpose(1, 2).reshape(
            B, queries.shape[1], L, self.max_width
        )

        if mask_invalid:
            logits = logits.masked_fill(~valid[:, None, :, :], float("-inf"))
        return logits


class SpanRepLayer(nn.Module):
    """Factory class for various span representation approaches.

    This class provides a unified interface to instantiate different span
    representation methods based on the specified mode.

    Attributes:
        span_rep_layer (nn.Module): The underlying span representation layer.
    """

    def __init__(self, hidden_size, max_width, span_mode, **kwargs):
        """Initialize the SpanRepLayer with the specified mode.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
            span_mode (str): Type of span representation to use. Options:
                - 'markerV0': SpanMarkerV0
            **kwargs: Additional arguments passed to the span representation layer.

        Raises:
            ValueError: If an unknown span_mode is provided.
        """
        super().__init__()

        if span_mode == "markerV0":
            self.span_rep_layer = SpanMarkerV0(hidden_size, max_width, **kwargs)
        else:
            raise ValueError(f"Unknown span mode {span_mode}")

    def forward(self, x, *args):
        """Forward pass through the selected span representation layer.

        Args:
            x (torch.Tensor): Input tensor, typically of shape [B, L, D].
            *args: Additional arguments passed to the underlying layer.

        Returns:
            torch.Tensor: Span representations, typically of shape
                [B, L, max_width, D].
        """
        return self.span_rep_layer(x, *args)

    def score_queries(
        self,
        x: torch.Tensor,
        queries: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        *,
        mask_invalid: bool = True,
    ) -> torch.Tensor:
        """Direct factorized span scoring for ``markerV0``."""
        return self.span_rep_layer.score_queries(
            x, queries, text_lengths, mask_invalid=mask_invalid
        )
