#!/usr/bin/env python3
"""
hierarchical_clustering.py
--------------------------
Final stage of research pipeline: normalize evidence, filter codes, and run HCA.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Dict, List, Tuple, Any, Optional


def normalize_evidence(
    leaf_codes: List[Dict], total_articles: int
) -> Dict[int, np.ndarray]:
    """
    Convert each code's raw evidence into a presence vector of length total_articles.

    Args:
        leaf_codes: List of code dictionaries with evidence
        total_articles: Total number of articles in the dataset

    Returns:
        Dictionary mapping code_id to presence vector (1 if code appears, 0 otherwise)
    """
    norm_vectors = {}

    for code in leaf_codes:
        code_id = code["id"]
        evidence = code.get("evidence", {})

        # Create presence vector
        presence = np.zeros(total_articles, dtype=int)

        for article_id_str, quotes in evidence.items():
            article_id = int(article_id_str) - 1  # Convert to 0-based index
            if article_id < total_articles and len(quotes) > 0:
                presence[article_id] = 1

        norm_vectors[code_id] = presence

    return norm_vectors


def filter_low_freq(
    leaf_codes: List[Dict], norm_vectors: Dict[int, np.ndarray], min_pct: float = 0.05
) -> Tuple[List[Dict], Dict[int, np.ndarray]]:
    """
    Keep only codes that appear in >= min_pct of articles.

    Args:
        leaf_codes: List of code dictionaries
        norm_vectors: Dictionary of presence vectors
        min_pct: Minimum percentage of articles (default 5%)

    Returns:
        Tuple of (filtered_codes, filtered_vectors)
    """
    total_articles = len(next(iter(norm_vectors.values())))
    min_articles = int(total_articles * min_pct)

    filtered_codes = []
    filtered_vectors = {}

    for code in leaf_codes:
        code_id = code["id"]
        if code_id in norm_vectors:
            presence_count = np.sum(norm_vectors[code_id])
            if presence_count >= min_articles:
                filtered_codes.append(code)
                filtered_vectors[code_id] = norm_vectors[code_id]

    print(
        f"Filtered codes: {len(filtered_codes)} remaining (min_pct={min_pct}, min_articles={min_articles})"
    )
    if filtered_codes:
        evidence_counts = [
            sum(len(quotes) for quotes in code.get("evidence", {}).values())
            for code in filtered_codes
        ]
        avg_evidence = sum(evidence_counts) / len(filtered_codes)
        print(f"Average evidence per code: {avg_evidence:.2f}")
    else:
        print("Average evidence per code: 0.00")
    return filtered_codes, filtered_vectors


def build_feature_matrix(
    filtered_codes: List[Dict], filtered_vectors: Dict[int, np.ndarray]
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix for clustering.

    Args:
        filtered_codes: List of filtered code dictionaries
        filtered_vectors: Dictionary of presence vectors

    Returns:
        Tuple of (X matrix shape: num_codes × num_articles, labels list)
    """
    labels = []
    X_rows = []

    for code in filtered_codes:
        code_id = code["id"]
        labels.append(code["name"])
        X_rows.append(filtered_vectors[code_id])

    X = np.array(X_rows)
    return X, labels


def find_elbow(Z: np.ndarray, max_k: int = 10) -> int:
    """
    Find optimal number of clusters using elbow method.

    Args:
        Z: Linkage matrix
        max_k: Maximum number of clusters to consider

    Returns:
        Optimal number of clusters
    """
    distances = []
    K = range(1, min(max_k + 1, len(Z) + 1))

    for k in K:
        if k == 1:
            # Single cluster case
            distances.append(Z[-1, 2])
        else:
            # Get cluster assignments for k clusters
            clusters = fcluster(Z, k, criterion="maxclust")
            # Use the distance at which k clusters are formed
            merge_distances = Z[:, 2]
            # Distance when we have k clusters is at position (n-k)
            if len(Z) - k + 1 >= 0:
                distances.append(merge_distances[len(Z) - k])
            else:
                distances.append(0)

    # Find elbow using second derivative
    if len(distances) < 3:
        return 2  # Default to 2 clusters if not enough data

    # Calculate second derivatives
    second_derivatives = []
    for i in range(1, len(distances) - 1):
        second_deriv = distances[i - 1] - 2 * distances[i] + distances[i + 1]
        second_derivatives.append(second_deriv)

    # Find the point with maximum second derivative (sharpest bend)
    elbow_idx = np.argmax(second_derivatives) + 1  # +1 because we started from index 1
    optimal_k = K[elbow_idx]

    return optimal_k


def run_hca(
    X: np.ndarray,
    labels: List[str],
    method: str = "ward",
    metric: str = "euclidean",
    max_clusters: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run hierarchical clustering analysis with automatic cluster determination using elbow method.

    Args:
        X: Feature matrix (num_codes × num_articles)
        labels: List of code names
        method: Linkage method (default: ward)
        metric: Distance metric (default: euclidean)
        max_clusters: Maximum number of clusters to consider for elbow method

    Returns:
        Tuple of (linkage_matrix, cluster_assignments)
    """
    # Compute linkage matrix
    Z = linkage(X, method=method, metric=metric)

    # Find optimal number of clusters using elbow method
    optimal_k = find_elbow(Z, max_clusters)
    print(f"Elbow method determined optimal clusters: {optimal_k}")

    # Get cluster assignments using optimal k
    clusters = fcluster(Z, optimal_k, criterion="maxclust")

    return Z, clusters


def run_pipeline(leaf_codes: List[Dict], total_articles: int) -> Dict[str, Any]:
    """
    Execute the entire HCA pipeline end-to-end.

    Args:
        leaf_codes: List of code dictionaries with evidence
        total_articles: Total number of articles in dataset

    Returns:
        Dictionary containing clustering results
    """
    # Step 1: Normalize evidence
    norm_vectors = normalize_evidence(leaf_codes, total_articles)

    # Step 2: Filter low frequency codes
    filtered_codes, filtered_vectors = filter_low_freq(leaf_codes, norm_vectors)

    # Step 3: Build feature matrix
    X, labels = build_feature_matrix(filtered_codes, filtered_vectors)

    # Step 4: Run hierarchical clustering
    Z, clusters = run_hca(X, labels)

    return {
        "linkage_matrix": Z,
        "cluster_assignments": clusters,
        "labels": labels,
        "filtered_codes": filtered_codes,
        "feature_matrix": X,
    }


def generate_cluster_means_table(
    filtered_codes: List[Dict], clusters: np.ndarray, X: np.ndarray
) -> pd.DataFrame:
    """
    Generate cluster means table with individual codes (function - name) on y-axis and clusters on x-axis.

    Args:
        filtered_codes: List of filtered code dictionaries
        clusters: Cluster assignments for each code
        X: Feature matrix (num_codes × num_articles)

    Returns:
        DataFrame with individual codes as rows and cluster means as columns
    """
    unique_clusters = sorted(set(clusters))

    # Initialize results dictionary
    results = {}

    for i, code in enumerate(filtered_codes):
        # Create row label as "FUNCTION - Code Name"
        row_label = f"{code['function']} - {code['name']}"
        results[row_label] = {}

        # Get the cluster assignment for this code
        code_cluster = clusters[i]

        # Calculate mean presence for this code across all articles
        code_mean = np.mean(X[i])

        # Assign the mean to the appropriate cluster column, 0 to others
        for cluster_id in unique_clusters:
            if code_cluster == cluster_id:
                results[row_label][f"Cluster_{cluster_id}"] = code_mean
            else:
                results[row_label][f"Cluster_{cluster_id}"] = 0.0

    # Convert to DataFrame
    df = pd.DataFrame(results).T

    # Ensure all cluster columns exist and reorder
    cluster_cols = [f"Cluster_{i}" for i in unique_clusters]
    df = df[cluster_cols]

    # Sort by frame function order matching the current framework
    function_order = [
        "TOPIC",
        "BENEFIT_ATTRIBUTION",
        "RISK_ATTRIBUTION",
        "BENEFIT_EVALUATION",
        "RISK_EVALUATION",
        "TREATMENT",
    ]

    # Create a sorting key based on function order and then by code name
    def sort_key(row_label):
        function = row_label.split(" - ")[0]
        code_name = row_label.split(" - ")[1] if " - " in row_label else ""

        # Get function priority (lower number = higher priority)
        func_priority = (
            function_order.index(function) if function in function_order else 999
        )

        return (func_priority, code_name)

    # Sort the DataFrame by the custom key
    sorted_indices = sorted(df.index, key=sort_key)
    df = df.reindex(sorted_indices)

    return df


def save_cluster_means_csv(
    filtered_codes: List[Dict],
    clusters: np.ndarray,
    X: np.ndarray,
    filename: str = "cluster_means.csv",
):
    """
    Generate and save cluster means table as CSV.

    Args:
        filtered_codes: List of filtered code dictionaries
        clusters: Cluster assignments for each code
        X: Feature matrix (num_codes × num_articles)
        filename: Output CSV filename
    """
    df = generate_cluster_means_table(filtered_codes, clusters, X)
    df.to_csv(filename)
    print(f"Cluster means table saved to: {filename}")


def infer_total_articles(codes: List[Dict[str, Any]], fallback: int = 50) -> int:
    """Infer total articles from evidence keys; fallback to provided default."""
    max_id = 0
    for code in codes:
        evidence = code.get("evidence", {}) or {}
        for aid in evidence.keys():
            try:
                max_id = max(max_id, int(aid))
            except (TypeError, ValueError):
                continue
    return max_id if max_id > 0 else fallback


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hierarchical clustering on a codebook and output cluster means CSV."
    )
    parser.add_argument(
        "--model",
        choices=["gpt", "gemini"],
        help="Shortcut to select default codebook paths and filenames.",
    )
    parser.add_argument(
        "--codebook",
        help="Path to codebook JSON. Overrides --model defaults when provided.",
    )
    parser.add_argument(
        "--output",
        help="Output CSV filename for cluster means. Defaults to cluster_means_<model>.csv or cluster_means.csv.",
    )
    parser.add_argument(
        "--total-articles",
        type=int,
        dest="total_articles",
        help="Override inferred total article count.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    """
    Main function to run the hierarchical clustering pipeline with optional model/codebook/output overrides.
    """
    import sys
    import os

    args = parse_args(argv)

    # Import helpers
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.get_root_codes import get_root_codes

    # Resolve codebook path via model shortcut or explicit path
    default_codebooks = {
        "gpt": "gpt_4o_mini_codebook.json",
        "gemini": "gemini_2_0_flash_lite_codebook.json",
    }
    model_key = args.model or "gpt"
    codebook_path = args.codebook or default_codebooks.get(model_key, "gpt_4o_mini_codebook.json")

    # Resolve output filename
    default_output = f"cluster_means_{model_key}.csv" if args.model else "cluster_means.csv"
    output_path = args.output or default_output

    try:
        print(f"Loading leaf/root codes from: {codebook_path}")
        leaf_codes = get_root_codes(codebook_path)
        print(f"Found {len(leaf_codes)} codes")

        # Infer total articles unless overridden
        total_articles = args.total_articles or infer_total_articles(leaf_codes)
        if args.total_articles is None:
            print(f"Inferred total articles: {total_articles}")

        print("Running hierarchical clustering pipeline...")
        results = run_pipeline(leaf_codes, total_articles)

        print("Clustering complete:")
        print(f"  - {len(results['filtered_codes'])} codes after filtering")
        print(f"  - {len(set(results['cluster_assignments']))} clusters identified")

        # Save cluster means table
        print(f"Generating cluster means table -> {output_path}")
        save_cluster_means_csv(
            filtered_codes=results["filtered_codes"],
            clusters=results["cluster_assignments"],
            X=results["feature_matrix"],
            filename=output_path,
        )

        print("Pipeline completed successfully!")

    except FileNotFoundError:
        print(f"Error: Could not find codebook file: {codebook_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
