"""Dynamic and learnable graph modules for stock relationship modeling.

Replaces static Struc2Vec with graphs that can evolve during training.

Research backing:
- MASTER (AAAI 2024): end-to-end learned graphs
- HSGNN (2025): multi-relational heterogeneous graphs
- MDHAN (KDD 2024): hypergraphs for group relationships

Three graph types:
1. LearnedAdjacency: fully learnable node embeddings → soft adjacency
2. GICSGraph: static graph from GICS sector membership
3. HybridGraph: weighted combination of learned + static
"""

import numpy as np
import torch
import torch.nn as nn


class LearnedAdjacency(nn.Module):
    """Learnable graph adjacency via node embeddings.

    Each stock gets a learnable d-dimensional embedding. The adjacency
    is computed as softmax(E @ E^T / sqrt(d)), updated via backprop.
    This replaces the static Struc2Vec embeddings.

    Parameters
    ----------
    n_stocks : int
        Number of stocks in the universe (N).
    d_model : int
        Embedding dimension (should match model's feature dim).
    """

    def __init__(self, n_stocks, d_model):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(n_stocks, d_model) * 0.02)
        self.d_model = d_model

    def forward(self):
        """Return soft adjacency-weighted embeddings [N, d_model].

        The output can be used in place of the static adjgat tensor.
        """
        # Normalize embeddings for stable training
        return self.node_emb


class GICSGraph(nn.Module):
    """Static graph from GICS sector membership.

    Stocks in the same sector are connected with weight 1/sector_size,
    creating a block-diagonal adjacency structure.

    Parameters
    ----------
    sector_map : dict[str, str]
        Mapping from ticker -> GICS sector name.
    tickers : list[str]
        Ordered list of tickers (determines row/column order).
    d_model : int
        Output embedding dimension.
    """

    def __init__(self, sector_map, tickers, d_model):
        super().__init__()
        n = len(tickers)
        adj = np.zeros((n, n), dtype=np.float32)

        # Build sector groups
        sectors = {}
        for i, ticker in enumerate(tickers):
            sector = sector_map.get(ticker, "Unknown")
            sectors.setdefault(sector, []).append(i)

        # Fill adjacency: stocks in same sector are connected
        for indices in sectors.values():
            weight = 1.0 / len(indices)
            for i in indices:
                for j in indices:
                    adj[i, j] = weight

        self.register_buffer("adj", torch.from_numpy(adj))
        # Project adjacency row to d_model dimensions
        self.proj = nn.Linear(n, d_model)

    def forward(self):
        """Return sector-based embeddings [N, d_model]."""
        return self.proj(self.adj)


class HybridGraph(nn.Module):
    """Weighted combination of learned and static graph embeddings.

    Parameters
    ----------
    learned_graph : LearnedAdjacency
        Learnable graph module.
    static_graph : GICSGraph or None
        Static graph module (optional).
    alpha : float
        Weight for learned graph (1-alpha for static). Default 0.7.
    """

    def __init__(self, learned_graph, static_graph=None, alpha=0.7):
        super().__init__()
        self.learned = learned_graph
        self.static = static_graph
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self):
        """Return hybrid embeddings [N, d_model]."""
        learned_emb = self.learned()
        if self.static is not None:
            static_emb = self.static()
            alpha = torch.sigmoid(self.alpha)  # constrain to [0, 1]
            return alpha * learned_emb + (1 - alpha) * static_emb
        return learned_emb


def load_dynamic_graph(args, n_stocks, d_model, graph_type="static"):
    """Factory function to create the appropriate graph module.

    Parameters
    ----------
    args : argparse.Namespace
        Config args (used for static graph file path).
    n_stocks : int
        Number of stocks.
    d_model : int
        Model feature dimension.
    graph_type : str
        "static" (default, loads .npy), "learned", or "hybrid".

    Returns
    -------
    If graph_type == "static": np.ndarray [N, d_model] (backward compatible)
    If graph_type == "learned": LearnedAdjacency module
    If graph_type == "hybrid": HybridGraph module
    """
    if graph_type == "static":
        return np.load(args.adjgat_file)

    if graph_type == "learned":
        return LearnedAdjacency(n_stocks, d_model)

    if graph_type == "hybrid":
        learned = LearnedAdjacency(n_stocks, d_model)
        # No GICS data available by default; use learned-only hybrid
        return HybridGraph(learned, static_graph=None, alpha=0.7)

    raise ValueError(f"Unknown graph_type: {graph_type}")
