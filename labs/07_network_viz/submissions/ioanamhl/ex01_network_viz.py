"""
Exercise 8 — Visualization of Co-Expression Networks + Hub Genes

TODO:
- Load the expression matrix and module mapping from Lab 6
- Rebuild (or load) the adjacency matrix
- Construct the graph from adjacency
- Color nodes by module
- Compute hub genes (top degree)
- Visualize and export the network figure (.png)
- Export hub genes to CSV (optional)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# --------------------------
# Config — complete with your values
# --------------------------
HANDLE = "ioanamhl"

# Input files
EXPR_CSV = Path(f"data/work/{HANDLE}/lab06/expression_matrix2.csv")
MODULES_CSV = Path(f"labs/06_wgcna/submissions/{HANDLE}/modules_{HANDLE}.csv")

# Optional: if you saved adjacency in Lab 6, load it here
PRECOMPUTED_ADJ_CSV: Optional[Path] = None

# Parameters for adjacency reconstruction (if PRECOMPUTED_ADJ_CSV is None)
CORR_METHOD = "spearman"   # TODO: choose "pearson" or "spearman"
USE_ABS_CORR = True        # TODO: use absolute correlations?
ADJ_THRESHOLD = 0.6        # TODO: correlation threshold
WEIGHTED = False           # TODO: use weighted adjacency or binary?

# Visualization parameters
SEED = 42
TOPK_HUBS = 10
NODE_BASE_SIZE = 60
EDGE_ALPHA = 0.15

# Outputs
OUT_DIR = Path(f"labs/07_network_viz/submissions/{HANDLE}")
OUT_PNG = OUT_DIR / f"network_{HANDLE}.png"
OUT_HUBS = OUT_DIR / f"hubs_{HANDLE}.csv"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Nu am găsit: {path}")


def read_expression_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if df.empty:
        raise ValueError("Matricea de expresie este goală.")
    return df


def read_modules_csv(path: Path) -> Dict[str, int]:
    df = pd.read_csv(path)
    if not {"Gene", "Module"}.issubset(df.columns):
        raise ValueError("modules.csv trebuie să conțină coloanele: Gene, Module")
    return dict(zip(df["Gene"].astype(str), df["Module"].astype(int)))


def correlation_to_adjacency(expr: pd.DataFrame,
                             method: str,
                             use_abs: bool,
                             threshold: float,
                             weighted: bool) -> pd.DataFrame:
    corr = expr.T.corr(method=method)
    if use_abs:
        corr = corr.abs()
    if weighted:
        A = corr.copy()
        A[A < threshold] = 0.0
    else:
        A = (corr >= threshold).astype(float)
    np.fill_diagonal(A.values, 0.0)
    return A


def graph_from_adjacency(A: pd.DataFrame) -> nx.Graph:
    G = nx.from_pandas_adjacency(A)  
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
    return G


def color_map_from_modules(nodes: Iterable[str], gene2module: Dict[str, int]) -> Dict[str, str]:
    cmap = plt.get_cmap("tab10")
    colors: Dict[str, str] = {}
    for n in nodes:
        m = gene2module.get(n, 0)
        colors[n] = cmap((m - 1) % 10) if m > 0 else "#CCCCCC"
    return colors


def compute_hubs(G: nx.Graph, topk: int) -> pd.DataFrame:
    deg = dict(G.degree())
    btw = nx.betweenness_centrality(G, normalized=True, seed=SEED) if G.number_of_nodes() <= 5000 else {n: np.nan for n in G.nodes()}  
    hubs = (
        pd.DataFrame({"Gene": list(deg.keys()), "Degree": list(deg.values()), "Betweenness": [btw.get(n, np.nan) for n in deg.keys()]})
        .sort_values(["Degree", "Betweenness"], ascending=False)
        .head(topk)
        .reset_index(drop=True)
    )
    return hubs


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # 1) Verificări & încărcări
    ensure_exists(EXPR_CSV)
    ensure_exists(MODULES_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    expr = read_expression_matrix(EXPR_CSV)
    gene2module = read_modules_csv(MODULES_CSV)

    # 2) Adiacență: încărcați dacă aveți, altfel reconstruiți din corelații
    if PRECOMPUTED_ADJ_CSV is not None:
        ensure_exists(PRECOMPUTED_ADJ_CSV)
        A = pd.read_csv(PRECOMPUTED_ADJ_CSV, index_col=0)
        common = sorted(set(A.index) & set(gene2module.keys()))
        A = A.loc[common, common]
    else:
        A = correlation_to_adjacency(expr, CORR_METHOD, USE_ABS_CORR, ADJ_THRESHOLD, WEIGHTED)
        common = sorted(set(A.index) & set(gene2module.keys()))
        A = A.loc[common, common]

    # 3) Graf
    G = graph_from_adjacency(A)
    print(f"Grafic: {G.number_of_nodes()} noduri, {G.number_of_edges()} muchii.")

    # 4) Culori după modul
    node_colors_map = color_map_from_modules(G.nodes(), gene2module)
    node_colors = [node_colors_map[n] for n in G.nodes()]

    # 5) Hub genes
    hubs_df = compute_hubs(G, TOPK_HUBS)
    hubs_set = set(hubs_df["Gene"])
    node_sizes = [NODE_BASE_SIZE * (1.5 if n in hubs_set else 1.0) for n in G.nodes()]

    # 6) Layout + desen
    pos = nx.spring_layout(G, seed=SEED, k=None)  # k=None → auto; creșteți dacă doriți graf mai "întins"
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, alpha=EDGE_ALPHA, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, linewidths=0.0)

    # etichete doar pentru hub-uri
    hub_labels = {g: g for g in hubs_set if g in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=hub_labels, font_size=8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()
    print(f"Am salvat figura în: {OUT_PNG}")

    # 7) Export hub-uri
    hubs_df.to_csv(OUT_HUBS, index=False)
    print(f"Am salvat hub genes în: {OUT_HUBS}")