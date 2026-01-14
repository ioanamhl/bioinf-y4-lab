"""
Exercise 9.1 — Drug–Gene Bipartite Network & Drug Similarity Network

Scop:
- să construiți o rețea bipartită drug–gene plecând de la un CSV
- să proiectați layer-ul de medicamente folosind similaritatea dintre seturile de gene
- să exportați un fișier cu muchiile de similaritate între medicamente

TODO:
- încărcați datele drug–gene
- construiți dict-ul drug -> set de gene țintă
- construiți graful bipartit drug–gene (NetworkX)
- calculați similaritatea dintre medicamente (ex. Jaccard)
- construiți graful drug similarity
- exportați tabelul cu muchii: drug1, drug2, weight
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, List

import itertools
import pickle

import networkx as nx
import pandas as pd

# --------------------------
# Config — adaptați pentru handle-ul vostru
# --------------------------
HANDLE = "ioanamhl"  # 

# Input: fișier cu coloane cel puțin: drug, gene
DRUG_GENE_CSV = Path(f"data/work/{HANDLE}/lab09/drug_gene_{HANDLE}.csv")

# Output directory & files
OUT_DIR = Path(f"labs/09_repurposing/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DRUG_SUMMARY = OUT_DIR / f"drug_summary_{HANDLE}.csv"
OUT_DRUG_SIMILARITY = OUT_DIR / f"drug_similarity_{HANDLE}.csv"
OUT_GRAPH_DRUG_GENE = OUT_DIR / f"network_drug_gene_{HANDLE}.gpickle"


def ensure_exists(path: Path) -> None:
    """
    - verificați că fișierul există
    - dacă nu, ridicați FileNotFoundError cu un mesaj clar
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[ERROR] Nu găsesc fișierul de input: {path}\n"
            f"Verifică HANDLE-ul și calea DRUG_GENE_CSV."
        )
    if not path.is_file():
        raise FileNotFoundError(f"[ERROR] Calea există dar nu e fișier: {path}")


def load_drug_gene_table(path: Path) -> pd.DataFrame:
    """
    - citiți CSV-ul cu pandas
    - validați că există cel puțin coloanele: 'drug', 'gene'
    - returnați DataFrame-ul
    """
    df = pd.read_csv(path)

    # suportă și cazuri cu alte litere (Drug/Gene)
    cols_lower = {c.lower(): c for c in df.columns}
    if "drug" not in cols_lower or "gene" not in cols_lower:
        raise ValueError(
            f"[ERROR] CSV-ul trebuie să aibă coloanele 'drug' și 'gene'. "
            f"Am găsit: {list(df.columns)}"
        )

    df = df.rename(columns={cols_lower["drug"]: "drug", cols_lower["gene"]: "gene"})

    # curățare minimă
    df["drug"] = df["drug"].astype(str).str.strip()
    df["gene"] = df["gene"].astype(str).str.strip()
    df = df.dropna(subset=["drug", "gene"])
    df = df[(df["drug"] != "") & (df["gene"] != "")]
    df = df.drop_duplicates(subset=["drug", "gene"]).reset_index(drop=True)

    return df


def build_drug2genes(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    - construiți un dict: drug -> set de gene țintă
    - sugestie: folosiți groupby("drug") și aplicați set() pe coloana gene
    """
    drug2genes = (
        df.groupby("drug")["gene"]
        .apply(lambda s: set(s.dropna().astype(str).str.strip()))
        .to_dict()
    )
    # elimină seturi goale
    return {d: gs for d, gs in drug2genes.items() if gs}


def build_bipartite_graph(drug2genes: Dict[str, Set[str]]) -> nx.Graph:
    """
    - construiți graful bipartit:
      - nodurile 'drug' cu atribut bipartite="drug"
      - nodurile 'gene' cu atribut bipartite="gene"
      - muchii drug-gene
    """
    G = nx.Graph()

    for drug in drug2genes:
        G.add_node(drug, bipartite="drug")

    for drug, genes in drug2genes.items():
        for gene in genes:
            if not G.has_node(gene):
                G.add_node(gene, bipartite="gene")
            G.add_edge(drug, gene)

    return G


def summarize_drugs(drug2genes: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    - construiți un DataFrame cu:
        drug, num_targets (numărul de gene țintă)
    """
    rows = [{"drug": d, "num_targets": len(gs)} for d, gs in drug2genes.items()]
    out = pd.DataFrame(rows).sort_values(["num_targets", "drug"], ascending=[False, True])
    return out.reset_index(drop=True)


def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    """
    Calculați similaritatea Jaccard între două seturi de gene:
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not s1 and not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def compute_drug_similarity_edges(
    drug2genes: Dict[str, Set[str]],
    min_sim: float = 0.0,
) -> List[Tuple[str, str, float]]:
    """
    - pentru toate perechile de medicamente (combinații de câte 2),
      calculați similaritatea Jaccard între seturile de gene
    - rețineți doar muchiile cu similaritate >= min_sim
    - returnați o listă de tuple (drug1, drug2, weight)
    """
    drugs = sorted(drug2genes.keys())
    edges: List[Tuple[str, str, float]] = []

    for d1, d2 in itertools.combinations(drugs, 2):
        sim = jaccard_similarity(drug2genes[d1], drug2genes[d2])
        if sim >= min_sim:
            edges.append((d1, d2, float(sim)))

    edges.sort(key=lambda x: x[2], reverse=True)
    return edges


def edges_to_dataframe(edges: List[Tuple[str, str, float]]) -> pd.DataFrame:
    """
    - transformați lista de muchii (drug1, drug2, weight) într-un DataFrame
      cu coloanele: drug1, drug2, similarity
    """
    return pd.DataFrame(edges, columns=["drug1", "drug2", "similarity"])


def save_graph_pickle(G: nx.Graph, path: Path) -> None:
    """
    În unele versiuni NetworkX nu mai există nx.write_gpickle.
    Salvăm robust folosind pickle standard.
    """
    with open(path, "wb") as f:
        pickle.dump(G, f)


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # TODO 1: verificați că fișierul de input există
    ensure_exists(DRUG_GENE_CSV)

    # TODO 2: încărcați tabelul drug-gene
    df = load_drug_gene_table(DRUG_GENE_CSV)

    # TODO 3: construiți mapping-ul drug -> set de gene
    drug2genes = build_drug2genes(df)

    # TODO 4: construiți graful bipartit și salvați-l (opțional)
    G_bip = build_bipartite_graph(drug2genes)
    save_graph_pickle(G_bip, OUT_GRAPH_DRUG_GENE)

    # TODO 5: generați și salvați sumarul pe medicamente
    df_sum = summarize_drugs(drug2genes)
    df_sum.to_csv(OUT_DRUG_SUMMARY, index=False)

    # TODO 6: calculați similaritatea între medicamente
    # Poți seta un prag (ex: 0.1 / 0.2) ca să nu iasă un graf foarte dens
    edges = compute_drug_similarity_edges(drug2genes, min_sim=0.0)
    df_edges = edges_to_dataframe(edges)
    df_edges.to_csv(OUT_DRUG_SIMILARITY, index=False)

    print("[INFO] Done.")
    print(f"[INFO] Input rows: {len(df)}")
    print(f"[INFO] Drugs: {len(drug2genes)} | Genes: {df['gene'].nunique()}")
    print(f"[INFO] Bipartite graph: {G_bip.number_of_nodes()} nodes, {G_bip.number_of_edges()} edges")
    print(f"[INFO] Similarity edges saved: {len(df_edges)}")
    print(f"[INFO] Wrote:\n - {OUT_DRUG_SUMMARY}\n - {OUT_DRUG_SIMILARITY}\n - {OUT_GRAPH_DRUG_GENE}")
