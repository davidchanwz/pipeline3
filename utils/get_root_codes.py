"""
get_root_codes.py
----------------
Extracts all root codes (codes with no parent) from a codebook and compiles evidence from their children.

Usage:
    python utils/get_root_codes.py <codebook_json>
"""

import sys
import json
from collections import defaultdict

#!/usr/bin/env python3
"""
get_root_codes.py
----------------
Extract root codes from a codebook as a clean list of dictionaries.

Usage:
    python utils/get_root_codes.py [codebook_path]
"""

import json
import sys
from typing import Dict, List


def get_root_codes(codebook_path: str) -> List[Dict]:
    """
    Extract root codes from a codebook file.

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

    # Extract root codes with clean format (no parent)
    root_codes = []
    for code in codes:
        if code.get("parent_code_id") is None:
            root_codes.append(
                {
                    "id": code["code_id"],
                    "name": code.get("name", ""),
                    "function": code.get("function", ""),
                    "description": code.get("description", ""),
                    "evidence": code.get("evidence", {}),
                }
            )

    return root_codes


def main():
    codebook_path = sys.argv[1] if len(sys.argv) > 1 else "gpt_4o_mini_codebook.json"

    try:
        root_codes = get_root_codes(codebook_path)
        print(json.dumps(root_codes, indent=2))

    except FileNotFoundError:
        print(f"Error: Could not find codebook file: {codebook_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
