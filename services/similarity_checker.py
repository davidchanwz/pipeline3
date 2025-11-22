#!/usr/bin/env python3
"""
similarity_checker.py
--------------------
Simple service to compare leaf codes between two codebooks using cosine and Jaccard similarity.

Usage:
    python services/similarity_checker.py [--codebook1 path] [--codebook2 path]
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_codebook(filepath: str) -> Dict:
    """Load a codebook JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_leaf_codes(codebook: Dict) -> List[Dict]:
    """Extract only leaf codes (codes with no children)."""
    codes = codebook.get("codes", [])

    # Find all parent IDs
    parent_ids = set()
    for code in codes:
        if code.get("parent_code_id"):
            parent_ids.add(code["parent_code_id"])

    # Return codes that are not parents (leaf codes)
    leaf_codes = [code for code in codes if code["code_id"] not in parent_ids]
    return leaf_codes


def get_code_text(code: Dict) -> str:
    """Extract text content from a code for similarity comparison."""
    name = code.get("name", "")
    description = code.get("description", "")

    # Combine evidence quotes
    evidence_text = ""
    evidence = code.get("evidence", {})
    for article_id, quotes in evidence.items():
        if isinstance(quotes, list):
            evidence_text += " ".join(quotes)

    return f"{name} {description} {evidence_text}".strip()


def calculate_cosine_similarity(texts1: List[str], texts2: List[str]) -> np.ndarray:
    """Calculate cosine similarity between two sets of texts."""
    all_texts = texts1 + texts2

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Split back into two groups
    matrix1 = tfidf_matrix[: len(texts1)]
    matrix2 = tfidf_matrix[len(texts1) :]

    # Calculate cosine similarity
    return cosine_similarity(matrix1, matrix2)


def get_word_set(text: str) -> Set[str]:
    """Extract word set from text for Jaccard similarity."""
    words = text.lower().split()
    # Simple word cleaning
    cleaned_words = set()
    for word in words:
        # Remove punctuation and keep only alphabetic characters
        clean_word = "".join(c for c in word if c.isalpha())
        if len(clean_word) > 2:  # Keep words longer than 2 characters
            cleaned_words.add(clean_word)
    return cleaned_words


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def find_best_matches(codebook1: Dict, codebook2: Dict) -> Dict:
    """Find best matches between leaf codes of two codebooks."""
    leaf_codes1 = extract_leaf_codes(codebook1)
    leaf_codes2 = extract_leaf_codes(codebook2)

    print(f"Codebook 1: {len(leaf_codes1)} leaf codes")
    print(f"Codebook 2: {len(leaf_codes2)} leaf codes")

    # Extract text for each code
    texts1 = [get_code_text(code) for code in leaf_codes1]
    texts2 = [get_code_text(code) for code in leaf_codes2]

    # Calculate cosine similarity matrix
    cosine_matrix = calculate_cosine_similarity(texts1, texts2)

    # Find best matches
    matches = []

    for i, code1 in enumerate(leaf_codes1):
        best_cosine_idx = np.argmax(cosine_matrix[i])
        best_cosine_score = cosine_matrix[i][best_cosine_idx]

        # Calculate Jaccard similarity for the best match
        words1 = get_word_set(texts1[i])
        words2 = get_word_set(texts2[best_cosine_idx])
        jaccard_score = jaccard_similarity(words1, words2)

        matches.append(
            {
                "code1": {
                    "id": code1["code_id"],
                    "name": code1["name"],
                    "function": code1.get("function", ""),
                    "description": (
                        code1.get("description", "")[:100] + "..."
                        if len(code1.get("description", "")) > 100
                        else code1.get("description", "")
                    ),
                },
                "code2": {
                    "id": leaf_codes2[best_cosine_idx]["code_id"],
                    "name": leaf_codes2[best_cosine_idx]["name"],
                    "function": leaf_codes2[best_cosine_idx].get("function", ""),
                    "description": (
                        leaf_codes2[best_cosine_idx].get("description", "")[:100]
                        + "..."
                        if len(leaf_codes2[best_cosine_idx].get("description", ""))
                        > 100
                        else leaf_codes2[best_cosine_idx].get("description", "")
                    ),
                },
                "cosine_similarity": float(best_cosine_score),
                "jaccard_similarity": float(jaccard_score),
            }
        )

    return {"total_comparisons": len(matches), "matches": matches}


def print_results(results: Dict, codebook1_path: str, codebook2_path: str):
    """Print comparison results in a readable format."""
    print(f"\n" + "=" * 80)
    print(f"CODEBOOK SIMILARITY COMPARISON")
    print(f"=" * 80)
    print(f"Codebook 1: {codebook1_path}")
    print(f"Codebook 2: {codebook2_path}")
    print(f"Total leaf code comparisons: {results['total_comparisons']}")

    # Sort matches by cosine similarity (descending)
    sorted_matches = sorted(
        results["matches"], key=lambda x: x["cosine_similarity"], reverse=True
    )

    # Calculate average similarities
    avg_cosine = np.mean([m["cosine_similarity"] for m in sorted_matches])
    avg_jaccard = np.mean([m["jaccard_similarity"] for m in sorted_matches])

    print(f"\nAverage Cosine Similarity: {avg_cosine:.3f}")
    print(f"Average Jaccard Similarity: {avg_jaccard:.3f}")

    # Show top matches
    print(f"\nTOP 10 MATCHES (by cosine similarity):")
    print(
        f"{'Rank':<4} {'Cosine':<7} {'Jaccard':<8} {'Code1':<25} {'Code2':<25} {'Function Match'}"
    )
    print("-" * 95)

    for i, match in enumerate(sorted_matches[:10], 1):
        func_match = (
            "✓" if match["code1"]["function"] == match["code2"]["function"] else "✗"
        )
        print(
            f"{i:<4} {match['cosine_similarity']:<7.3f} {match['jaccard_similarity']:<8.3f} "
            f"{match['code1']['name'][:24]:<25} {match['code2']['name'][:24]:<25} {func_match}"
        )

    # Show function distribution
    print(f"\nFUNCTION MATCHING:")
    function_matches = sum(
        1 for m in sorted_matches if m["code1"]["function"] == m["code2"]["function"]
    )
    print(
        f"Same function: {function_matches}/{len(sorted_matches)} ({function_matches/len(sorted_matches)*100:.1f}%)"
    )

    # Show high similarity matches (>0.5 cosine)
    high_sim_matches = [m for m in sorted_matches if m["cosine_similarity"] > 0.5]
    print(
        f"High similarity matches (>0.5): {len(high_sim_matches)}/{len(sorted_matches)} ({len(high_sim_matches)/len(sorted_matches)*100:.1f}%)"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare leaf codes between two codebooks using cosine and Jaccard similarity"
    )
    parser.add_argument(
        "--codebook1",
        default="gpt_4o_mini_codebook.json",
        help="Path to first codebook (default: gpt_4o_mini_codebook.json)",
    )
    parser.add_argument(
        "--codebook2",
        default="gemini_2_0_flash_lite_codebook.json",
        help="Path to second codebook (default: gemini_2_0_flash_lite_codebook.json)",
    )

    args = parser.parse_args()

    try:
        # Load codebooks
        print(f"Loading codebooks...")
        codebook1 = load_codebook(args.codebook1)
        codebook2 = load_codebook(args.codebook2)

        # Compare codebooks
        print(f"Comparing leaf codes...")
        results = find_best_matches(codebook1, codebook2)

        # Print results
        print_results(results, args.codebook1, args.codebook2)

    except FileNotFoundError as e:
        print(f"Error: Could not find codebook file - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
