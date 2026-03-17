"""
Struc2Vec graph embedding for the S&P500 correlation graph.

Wraps the existing Stockformer preprocessing pipeline by:
  1. build_correlation_graph(): reading label.csv, computing |corr|>threshold edges,
     saving corr_adj.npy and data.edgelist.
  2. run_struc2vec(): training Struc2Vec on the edgelist (via ge library) and saving
     128_corr_struc2vec_adjgat.npy with shape [N, embed_size].

Usage:
  python scripts/sp500_pipeline/graph_embedding.py \
      --data_dir ./data/Stock_SP500_2018-01-01_2024-01-01

Install GraphEmbedding before running:
  pip install git+https://github.com/shenweichen/GraphEmbedding.git
"""
import argparse
import os

import numpy as np
import pandas as pd


def build_correlation_graph(data_dir: str, threshold: float = 0.3) -> int:
    """
    Build filtered correlation graph and save corr_adj.npy + data.edgelist.

    Reads label.csv (T x N), computes the N x N Pearson correlation matrix,
    filters edges where |corr| > threshold, writes the surviving edges to
    data.edgelist, and saves the full matrix to corr_adj.npy.

    Returns the number of edges written to the edgelist.
    """
    label_path = os.path.join(data_dir, "label.csv")
    df = pd.read_csv(label_path, index_col=0)
    df.fillna(0, inplace=True)

    # Zero-variance guard — identical to the existing preprocessing script
    epsilon = 1e-10
    std_devs = np.std(df.values, axis=0)
    zero_var = std_devs < epsilon
    if zero_var.any():
        df.iloc[:, zero_var] = epsilon

    corr_matrix = np.corrcoef(df.values, rowvar=False)
    np.save(os.path.join(data_dir, "corr_adj.npy"), corr_matrix)

    # Build filtered edge list (upper triangle only — undirected)
    edge_list = []
    n = corr_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            weight = corr_matrix[i, j]
            if abs(weight) > threshold:
                edge_list.append((i, j, weight))

    edgelist_path = os.path.join(data_dir, "data.edgelist")
    with open(edgelist_path, "w") as f:
        for edge in edge_list:
            f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")

    print(f"Edge list saved: {len(edge_list)} edges (threshold |corr|>{threshold})")
    return len(edge_list)


def run_struc2vec(data_dir: str, embed_size: int = 128, workers: int = 4) -> None:
    """
    Train Struc2Vec on data.edgelist and save 128_corr_struc2vec_adjgat.npy.

    Requires the GraphEmbedding library:
      pip install git+https://github.com/shenweichen/GraphEmbedding.git

    Parameters
    ----------
    data_dir : str
        Directory containing data.edgelist (produced by build_correlation_graph).
    embed_size : int
        Embedding dimension; must match [param] dims in Multitask_SP500.conf (default 128).
    workers : int
        Parallel workers for Struc2Vec random walk simulation.
    """
    try:
        # Load Struc2Vec directly from its source file to bypass ge/__init__.py,
        # which imports LINE → deepctr (an unneeded transitive dependency).
        import importlib.util, sys as _sys
        _ge_init = importlib.util.find_spec("ge")
        if _ge_init is None:
            raise ImportError("ge package not found")
        import os as _os
        _ge_dir = _os.path.dirname(_ge_init.origin)
        _s2v_path = _os.path.join(_ge_dir, "models", "struc2vec.py")
        _spec = importlib.util.spec_from_file_location("ge.models.struc2vec", _s2v_path)
        _mod = importlib.util.module_from_spec(_spec)
        _sys.modules["ge.models.struc2vec"] = _mod
        _spec.loader.exec_module(_mod)
        Struc2Vec = _mod.Struc2Vec
    except (ImportError, AttributeError, FileNotFoundError):
        raise ImportError(
            "GraphEmbedding library not installed. Run:\n"
            "  pip install fastdtw gensim\n"
            "  pip install git+https://github.com/shenweichen/GraphEmbedding.git --no-deps"
        )

    import networkx as nx

    edgelist_path = os.path.join(data_dir, "data.edgelist")
    G = nx.read_edgelist(
        edgelist_path,
        create_using=nx.DiGraph(),
        nodetype=None,
        data=[("weight", float)],
    )

    model = Struc2Vec(G, num_walks=10, walk_length=80, workers=workers, verbose=40)
    model.train(embed_size=embed_size)
    embeddings = model.get_embeddings()

    # Sort by integer node id to produce a stable [N, embed_size] array
    n_nodes = max(int(k) for k in embeddings.keys()) + 1
    embedding_array = np.zeros((n_nodes, embed_size), dtype=np.float32)
    for node_id, vec in embeddings.items():
        embedding_array[int(node_id)] = vec

    out_path = os.path.join(data_dir, "128_corr_struc2vec_adjgat.npy")
    np.save(out_path, embedding_array)
    print(f"Embedding saved: shape {embedding_array.shape} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Struc2Vec graph embedding")
    parser.add_argument(
        "--data_dir",
        default="./data/Stock_SP500_2018-01-01_2024-01-01",
        help="Directory containing label.csv and where outputs will be written",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Correlation threshold for edge inclusion (default: 0.3)",
    )
    parser.add_argument(
        "--embed_size",
        type=int,
        default=128,
        help="Struc2Vec embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for Struc2Vec (default: 4)",
    )
    args = parser.parse_args()

    build_correlation_graph(args.data_dir, threshold=args.threshold)
    run_struc2vec(args.data_dir, embed_size=args.embed_size, workers=args.workers)


if __name__ == "__main__":
    main()
