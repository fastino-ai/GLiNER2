"""
Span representation layers for GLiNER2.

This module is an in-house replacement for the span representation code
previously sourced from the external `gliner` package
(gliner.modeling.span_rep and gliner.modeling.layers). Only pure PyTorch
is required — no dependency on the `gliner` PyPI package.

All span_mode variants from the original GLiNER codebase are included so
that weights from any GLiNER-family checkpoint load correctly, even though
GLiNER2 currently only uses `markerV0` by default.
"""

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Utility: projection layer builder
# ---------------------------------------------------------------------------


def create_projection_layer(
    hidden_size: int,
    dropout: float,
    out_dim: int = None,
) -> nn.Sequential:
    """Build a two-layer projection: Linear -> ReLU -> Dropout -> Linear.

    Args:
        hidden_size: Input dimensionality.
        dropout: Dropout probability between the two linear layers.
        out_dim: Output dimensionality. Defaults to hidden_size.

    Returns:
        An nn.Sequential projection layer.
    """
    if out_dim is None:
        out_dim = hidden_size

    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * 4, out_dim),
    )


# ---------------------------------------------------------------------------
# Utility: gather span start/end tokens from sequence
# ---------------------------------------------------------------------------


def extract_elements(sequence: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather token embeddings at specified positions.

    Args:
        sequence: Token embeddings, shape [B, L, D].
        indices: Positions to gather, shape [B, K].

    Returns:
        Gathered embeddings, shape [B, K, D].
    """
    B, L, D = sequence.shape
    expanded_indices = indices.unsqueeze(2).expand(-1, -1, D)
    return torch.gather(sequence, 1, expanded_indices)


# ---------------------------------------------------------------------------
# Span representation modules
# ---------------------------------------------------------------------------


class SpanQuery(nn.Module):
    """Span representation via learned per-width query vectors."""

    def __init__(self, hidden_size: int, max_width: int, trainable: bool = True):
        super().__init__()
        self.query_seg = nn.Parameter(torch.randn(hidden_size, max_width))
        nn.init.uniform_(self.query_seg, a=-1, b=1)
        if not trainable:
            self.query_seg.requires_grad = False
        self.project = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())

    def forward(self, h: torch.Tensor, *args) -> torch.Tensor:
        span_rep = torch.einsum("bld,ds->blsd", h, self.query_seg)
        return self.project(span_rep)


class SpanMLP(nn.Module):
    """Span representation via a single linear projection to all widths."""

    def __init__(self, hidden_size: int, max_width: int):
        super().__init__()
        self.mlp = nn.Linear(hidden_size, hidden_size * max_width)

    def forward(self, h: torch.Tensor, *args) -> torch.Tensor:
        B, L, D = h.size()
        span_rep = self.mlp(h).view(B, L, -1, D)
        return span_rep.relu()


class SpanCAT(nn.Module):
    """Span representation by concatenating token embeddings with width embeddings."""

    def __init__(self, hidden_size: int, max_width: int):
        super().__init__()
        self.max_width = max_width
        self.query_seg = nn.Parameter(torch.randn(128, max_width))
        self.project = nn.Sequential(
            nn.Linear(hidden_size + 128, hidden_size),
            nn.ReLU(),
        )

    def forward(self, h: torch.Tensor, *args) -> torch.Tensor:
        B, L, D = h.size()
        h_exp = h.view(B, L, 1, D).repeat(1, 1, self.max_width, 1)
        q = self.query_seg.view(1, 1, self.max_width, -1).repeat(B, L, 1, 1)
        return self.project(torch.cat([h_exp, q], dim=-1))


class SpanConvBlock(nn.Module):
    """Single convolutional block for span representation."""

    def __init__(self, hidden_size: int, kernel_size: int, span_mode: str = "conv_normal"):
        super().__init__()
        if span_mode == "conv_conv":
            self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size)
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")
        elif span_mode == "conv_max":
            self.conv = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        elif span_mode in ("conv_mean", "conv_sum"):
            self.conv = nn.AvgPool1d(kernel_size=kernel_size, stride=1)
        self.span_mode = span_mode
        self.pad = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("bld->bdl", x)
        if self.pad > 0:
            x = F.pad(x, (0, self.pad), "constant", 0)
        x = self.conv(x)
        if self.span_mode == "conv_sum":
            x = x * (self.pad + 1)
        return torch.einsum("bdl->bld", x)


class SpanConv(nn.Module):
    """Stacked convolutional span representations for each width."""

    def __init__(self, hidden_size: int, max_width: int, span_mode: str):
        super().__init__()
        kernels = [i + 2 for i in range(max_width - 1)]
        self.convs = nn.ModuleList(
            SpanConvBlock(hidden_size, k, span_mode) for k in kernels
        )
        self.project = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        span_reps = [x] + [conv(x) for conv in self.convs]
        return self.project(torch.stack(span_reps, dim=-2))


class SpanMarker(nn.Module):
    """Span representation from start/end endpoint projections (original)."""

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
        )
        self.project_end = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
        )
        self.out_project = nn.Linear(hidden_size * 2, hidden_size, bias=True)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        B, L, D = h.size()
        start_span_rep = extract_elements(self.project_start(h), span_idx[:, :, 0])
        end_span_rep = extract_elements(self.project_end(h), span_idx[:, :, 1])
        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()
        return self.out_project(cat).view(B, L, self.max_width, D)


class SpanMarkerV0(nn.Module):
    """Marks and projects span endpoints using a four-layer MLP (used by default in GLiNER2)."""

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)
        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        B, L, D = h.size()
        start_span_rep = extract_elements(self.project_start(h), span_idx[:, :, 0])
        end_span_rep = extract_elements(self.project_end(h), span_idx[:, :, 1])
        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()
        return self.out_project(cat).view(B, L, self.max_width, D)


class SpanMarkerV1(nn.Module):
    """Span marker augmented with an average-token context embedding."""

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)
        self.project_first = create_projection_layer(hidden_size, dropout)
        self.out_project = create_projection_layer(hidden_size * 3, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        B, L, D = h.size()
        start_span_rep = extract_elements(self.project_start(h), span_idx[..., 0])
        end_span_rep = extract_elements(self.project_end(h), span_idx[..., 1])
        avg_rep = torch.mean(h, dim=1).unsqueeze(1).expand_as(start_span_rep)
        span_feat = torch.cat((start_span_rep, end_span_rep, avg_rep), dim=-1).relu()
        return self.out_project(span_feat).view(B, L, self.max_width, D)


class ConvShare(nn.Module):
    """Span representation via shared convolution weights across widths."""

    def __init__(self, hidden_size: int, max_width: int):
        super().__init__()
        self.max_width = max_width
        self.conv_weigth = nn.Parameter(torch.randn(hidden_size, hidden_size, max_width))
        nn.init.kaiming_uniform_(self.conv_weigth, nonlinearity="relu")
        self.project = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        span_reps = []
        x = torch.einsum("bld->bdl", x)
        for i in range(self.max_width):
            x_i = F.pad(x, (0, i), "constant", 0)
            out_i = F.conv1d(x_i, self.conv_weigth[:, :, : i + 1])
            span_reps.append(out_i.transpose(-1, -2))
        return self.project(torch.stack(span_reps, dim=-2))


class ConvShareV2(nn.Module):
    """ConvShare with Xavier init and no post-projection."""

    def __init__(self, hidden_size: int, max_width: int):
        super().__init__()
        self.max_width = max_width
        self.conv_weigth = nn.Parameter(torch.randn(hidden_size, hidden_size, max_width))
        nn.init.xavier_normal_(self.conv_weigth)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        span_reps = []
        x = torch.einsum("bld->bdl", x)
        for i in range(self.max_width):
            x_i = F.pad(x, (0, i), "constant", 0)
            out_i = F.conv1d(x_i, self.conv_weigth[:, :, : i + 1])
            span_reps.append(out_i.transpose(-1, -2))
        return torch.stack(span_reps, dim=-2)


# ---------------------------------------------------------------------------
# Factory: SpanRepLayer
# ---------------------------------------------------------------------------

_SPAN_MODE_REGISTRY = {
    "marker": lambda hidden_size, max_width, **kw: SpanMarker(hidden_size, max_width, **kw),
    "markerV0": lambda hidden_size, max_width, **kw: SpanMarkerV0(hidden_size, max_width, **kw),
    "markerV1": lambda hidden_size, max_width, **kw: SpanMarkerV1(hidden_size, max_width, **kw),
    "query": lambda hidden_size, max_width, **kw: SpanQuery(hidden_size, max_width, trainable=True),
    "mlp": lambda hidden_size, max_width, **kw: SpanMLP(hidden_size, max_width),
    "cat": lambda hidden_size, max_width, **kw: SpanCAT(hidden_size, max_width),
    "conv_conv": lambda hidden_size, max_width, **kw: SpanConv(hidden_size, max_width, "conv_conv"),
    "conv_max": lambda hidden_size, max_width, **kw: SpanConv(hidden_size, max_width, "conv_max"),
    "conv_mean": lambda hidden_size, max_width, **kw: SpanConv(hidden_size, max_width, "conv_mean"),
    "conv_sum": lambda hidden_size, max_width, **kw: SpanConv(hidden_size, max_width, "conv_sum"),
    "conv_share": lambda hidden_size, max_width, **kw: ConvShare(hidden_size, max_width),
}


class SpanRepLayer(nn.Module):
    """Factory module that selects and wraps a span representation strategy.

    Args:
        hidden_size: Token embedding dimensionality.
        max_width: Maximum span width (in tokens).
        span_mode: One of the registered span representation modes.
        **kwargs: Forwarded to the selected sub-module (e.g., dropout).

    Raises:
        ValueError: If span_mode is not a registered mode.
    """

    def __init__(self, hidden_size: int, max_width: int, span_mode: str, **kwargs):
        super().__init__()
        if span_mode not in _SPAN_MODE_REGISTRY:
            raise ValueError(
                f"Unknown span mode '{span_mode}'. "
                f"Valid options: {sorted(_SPAN_MODE_REGISTRY)}"
            )
        self.span_rep_layer = _SPAN_MODE_REGISTRY[span_mode](hidden_size, max_width, **kwargs)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return self.span_rep_layer(x, *args)
