#!/usr/bin/env python3
import os
import argparse

import pandas as pd
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_embeddings(parquet_path: str, embedding_field: str):
    df = pd.read_parquet(parquet_path)
    if embedding_field not in df.columns:
        raise ValueError(f"Embedding column '{embedding_field}' not found")
    # Ensure they’re numpy arrays
    embeddings = np.vstack(df[embedding_field].values)
    return df, embeddings

def reduce_umap(embeddings: np.ndarray, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    return reducer.fit_transform(embeddings)

def cluster_hdbscan(emb_2d: np.ndarray, min_cluster_size=5, min_samples=1):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    return clusterer.fit_predict(emb_2d)

def plot_clusters(emb_2d: np.ndarray, labels: np.ndarray, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = cm.get_cmap('tab20', len(unique_labels))

    for lbl in unique_labels:
        mask = labels == lbl
        color = 'lightgrey' if lbl == -1 else colors(lbl)
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            s=10,
            c=[color],
            label=f"cluster {lbl}" if lbl != -1 else "noise",
            alpha=0.7
        )
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.title("UMAP projection of embeddings with HDBSCAN clustering")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved cluster plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize and cluster embeddings from a Parquet dataset"
    )
    parser.add_argument("--parquet", "-p", required=True,
                        help="Path to input Parquet file")
    parser.add_argument("--field", "-f", default="embedding",
                        help="Name of the embedding column (array-like)")
    parser.add_argument("--out", "-o", default="report/embeddings_umap.png",
                        help="Output plot path")
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1,
                        help="UMAP min_dist")
    parser.add_argument("--min_cluster_size", type=int, default=5,
                        help="HDBSCAN min_cluster_size")
    parser.add_argument("--min_samples", type=int, default=1,
                        help="HDBSCAN min_samples")
    args = parser.parse_args()

    print("🔍 Loading embeddings...")
    df, embeddings = load_embeddings(args.parquet, args.field)

    print("✨ Running UMAP reduction...")
    emb_2d = reduce_umap(
        embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )

    print("📦 Clustering with HDBSCAN...")
    labels = cluster_hdbscan(
        emb_2d,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples
    )

    print("🎨 Plotting clusters...")
    plot_clusters(emb_2d, labels, args.out)

if __name__ == "__main__":
    main()
