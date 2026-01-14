"""
Exercise 9.2 — Disease Proximity and Drug Ranking

Scop:
- să calculați distanța medie dintre fiecare medicament și un set de gene asociate unei boli
- să ordonați medicamentele în funcție de proximitate (network-based prioritization)

NOTE:
- graful bipartit din ex. 9.1 a fost salvat cu pickle (chiar dacă extensia e .gpickle).
  Aici îl încărcăm robust cu pickle.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, List

import pickle

import networkx as nx
import pandas as pd

# --------------------------
# Config
# --------------------------
HANDLE = "ioanamhl"  

# Input: graful bipartit (salvat anterior) SAU tabelul drug-gene
GRAPH_DRUG_GENE = Path(
    f"labs/09_repurposing/submissions/{HANDLE}/network_drug_gene_{HANDLE}.gpickle"
)
DRUG_GENE_CSV = Path(f"data/work/{HANDLE}/lab09/drug_gene_{HANDLE}.csv")

# Input: lista genelor bolii
DISEASE_GENES_TXT = Path(f"data/work/{HANDLE}/lab09/disease_genes_{HANDLE}.txt")

# Output directory & file
OUT_DIR = Path(f"labs/09_repurposing/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DRUG_PRIORITY = OUT_DIR / f"drug_priority_{HANDLE}.csv"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    """
    - verificați că fișierul există
    - dacă nu, ridicați FileNotFoundError
    """
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Nu găsesc fișierul: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"[ERROR] Calea există dar nu e fișier: {path}")


def load_drug_gene_table(path: Path) -> pd.DataFrame:
    """
    Citește CSV cu coloane drug, gene (case-insensitive) și curăță duplicate/NA.
    """
    df = pd.read_csv(path)

    cols_lower = {c.lower(): c for c in df.columns}
    if "drug" not in cols_lower or "gene" not in cols_lower:
        raise ValueError(
            f"[ERROR] CSV-ul trebuie să aibă coloanele 'drug' și 'gene'. "
            f"Am găsit: {list(df.columns)}"
        )

    df = df.rename(columns={cols_lower["drug"]: "drug", cols_lower["gene"]: "gene"})

    df["drug"] = df["drug"].astype(str).str.strip()
    df["gene"] = df["gene"].astype(str).str.strip()
    df = df.dropna(subset=["drug", "gene"])
    df = df[(df["drug"] != "") & (df["gene"] != "")]
    df = df.drop_duplicates(subset=["drug", "gene"]).reset_index(drop=True)

    return df


def build_drug2genes(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    drug -> set(genes)
    """
    drug2genes = (
        df.groupby("drug")["gene"]
        .apply(lambda s: set(s.dropna().astype(str).str.strip()))
        .to_dict()
    )
    return {d: gs for d, gs in drug2genes.items() if gs}


def build_bipartite_graph(drug2genes: Dict[str, Set[str]]) -> nx.Graph:
    """
    Graful bipartit drug–gene.
    """
    B = nx.Graph()

    for drug in drug2genes:
        B.add_node(drug, bipartite="drug")

    for drug, genes in drug2genes.items():
        for gene in genes:
            if not B.has_node(gene):
                B.add_node(gene, bipartite="gene")
            B.add_edge(drug, gene)

    return B


def load_bipartite_graph_or_build() -> nx.Graph:
    """
    - dacă GRAPH_DRUG_GENE există, încărcați-l direct
    - altfel, reconstruiți graful plecând de la DRUG_GENE_CSV
    """
    if GRAPH_DRUG_GENE.exists():
        # graful din ex 9.1 e salvat cu pickle
        with open(GRAPH_DRUG_GENE, "rb") as f:
            B = pickle.load(f)
        if not isinstance(B, nx.Graph):
            raise TypeError("[ERROR] Obiectul încărcat nu pare a fi un NetworkX Graph.")
        return B

    # fallback: build from CSV
    ensure_exists(DRUG_GENE_CSV)
    df = load_drug_gene_table(DRUG_GENE_CSV)
    drug2genes = build_drug2genes(df)
    return build_bipartite_graph(drug2genes)


def load_disease_genes(path: Path) -> Set[str]:
    """
    - încărcați fișierul text cu gene (una pe linie)
    - ignorați linii goale și comentarii (#)
    """
    ensure_exists(path)
    genes: Set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            genes.add(s)

    return genes


def get_drug_nodes(B: nx.Graph) -> List[str]:
    """
    Extrage nodurile de tip drug (bipartite="drug").
    """
    return [n for n, d in B.nodes(data=True) if d.get("bipartite") == "drug"]


def compute_drug_disease_distance(
    B: nx.Graph,
    drug: str,
    disease_genes: Set[str],
    mode: str = "mean",
    max_dist: int = 5,
) -> float:
    """
    Pentru un medicament:
    - calculează distanța shortest path către fiecare genă din disease_genes (care e în graf)
    - dacă nu există drum către o genă (deși e în graf), aplică penalizare max_dist+1
    - dacă boala nu are nicio genă prezentă în graf, returnează penalizare max_dist+1
    - mode: "mean" sau "min"
    """
    if drug not in B:
        return float(max_dist + 1)

    genes_in_graph = [g for g in disease_genes if g in B]
    if not genes_in_graph:
        return float(max_dist + 1)

    # Distanțe shortest path din drug către toate nodurile (BFS)
    lengths = nx.single_source_shortest_path_length(B, drug, cutoff=max_dist)

    dists: List[int] = []
    penalty = max_dist + 1

    for g in genes_in_graph:
        dists.append(lengths.get(g, penalty))

    if not dists:
        return float(penalty)

    mode = mode.lower().strip()
    if mode == "min":
        return float(min(dists))
    if mode == "mean":
        return float(sum(dists) / len(dists))

    raise ValueError("[ERROR] mode trebuie să fie 'mean' sau 'min'.")


def rank_drugs_by_proximity(
    B: nx.Graph,
    disease_genes: Set[str],
    mode: str = "mean",
) -> pd.DataFrame:
    """
    Pentru fiecare medicament:
    - calculează distance score
    - DataFrame: drug, distance
    - sort crescător (mai mic = mai aproape)
    """
    drugs = get_drug_nodes(B)

    rows = []
    for drug in drugs:
        dist = compute_drug_disease_distance(B, drug, disease_genes, mode=mode)
        rows.append({"drug": drug, "distance": dist})

    out = pd.DataFrame(rows).sort_values(["distance", "drug"], ascending=[True, True])
    return out.reset_index(drop=True)


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # TODO 1: verificați input-urile
    # (graful e opțional; dacă nu există, trebuie CSV-ul)
    if not GRAPH_DRUG_GENE.exists():
        ensure_exists(DRUG_GENE_CSV)
    ensure_exists(DISEASE_GENES_TXT)

    # TODO 2: încărcați / construiți graful bipartit
    B = load_bipartite_graph_or_build()

    # TODO 3: încărcați setul de disease genes
    disease_genes = load_disease_genes(DISEASE_GENES_TXT)

    # TODO 4: calculați ranking-ul medicamentelor după proximitate
    df_rank = rank_drugs_by_proximity(B, disease_genes, mode="mean")

    # TODO 5: salvați rezultatele
    df_rank.to_csv(OUT_DRUG_PRIORITY, index=False)

    print("[INFO] Done.")
    print(f"[INFO] Disease genes (total): {len(disease_genes)}")
    print(f"[INFO] Graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
    print(f"[INFO] Drugs ranked: {len(df_rank)}")
    print(f"[INFO] Wrote: {OUT_DRUG_PRIORITY}")
