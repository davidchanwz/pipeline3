#!/usr/bin/env python3
"""
get_leaf_codes.py
----------------
Extract leaf codes from a codebook as a clean list of dictionaries.

Usage:
    python utils/get_leaf_codes.py [codebook_path]
"""

import json
import sys
from typing import Dict, List


def get_leaf_codes(codebook_path: str) -> List[Dict]:
    """
    Extract leaf codes from a codebook file.

    Returns list of dicts with format:
    {
        "id": int,
        "name": str,
        "function": str,
        "description": str,
        "evidence": { article_id: [quotes...] }
    }
    """
    with open(codebook_path, "r") as f:
        codebook = json.load(f)

    codes = codebook.get("codes", [])

    # Find all parent code IDs
    parent_ids = set()
    for code in codes:
        parent_id = code.get("parent_code_id")
        if parent_id:
            parent_ids.add(parent_id)

    # Extract leaf codes with clean format
    leaf_codes = []
    for code in codes:
        if code["code_id"] not in parent_ids:
            leaf_codes.append(
                {
                    "id": code["code_id"],
                    "name": code.get("name", ""),
                    "function": code.get("function", ""),
                    "description": code.get("description", ""),
                    "evidence": code.get("evidence", {}),
                }
            )

    return leaf_codes


def main():
    """Main entry point."""
    codebook_path = sys.argv[1] if len(sys.argv) > 1 else "gpt_4o_mini_codebook.json"

    try:
        leaf_codes = get_leaf_codes(codebook_path)
        print(json.dumps(leaf_codes, indent=2))

    except FileNotFoundError:
        print(f"Error: Could not find codebook file: {codebook_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
