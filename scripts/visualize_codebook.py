#!/usr/bin/env python3
"""
visualize_codebook.py
---------------------
Generate a simple HTML visualization of a codebook JSON file, including merged-code provenance for root codes.

Usage:
    python scripts/visualize_codebook.py /path/to/codebook.json [output.html]

- The input JSON is expected to have a top-level "codes" array with objects containing at least:
  - code_id (int or str)
  - name (str)
  - parent_code_id (nullable int/str)
  - description (str)
  - evidence (dict mapping article_id -> list of quotes) OR evidence may be a list (legacy)

- The script outputs a single HTML file that shows the hierarchy (parents -> children)
  and per-code evidence grouped by article id. The HTML is compact and uses
  lightweight JS for collapsing sections.

Focus: correctness and efficiency; no external dependencies.
"""

import json
import os
import sys
import argparse
from typing import Dict, Any, List, Optional


def load_codebook(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_evidence(evidence: Any) -> Dict[str, List[str]]:
    """Normalize evidence into a dict mapping str(article_id) -> list[str].
    Accepts either a dict (possibly with int keys) or a list of quotes (legacy).
    """
    if evidence is None:
        return {}
    if isinstance(evidence, dict):
        normalized = {}
        for k, v in evidence.items():
            key = str(k)
            # v should be list of strings; if single string, wrap it
            if isinstance(v, list):
                normalized[key] = [str(x) for x in v]
            else:
                normalized[key] = [str(v)]
        return normalized
    # If it's a list, put under a placeholder article id
    if isinstance(evidence, list):
        return {"_quotes": [str(x) for x in evidence]}
    # Otherwise, fallback to string
    return {"_raw": [str(evidence)]}


def build_tree(codes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a tree of codes. Returns list of root nodes.
    Each node will be a dict with keys: code (original), children (list).
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    for c in codes:
        cid = str(c.get("code_id") or c.get("id"))
        nodes[cid] = {"code": c, "children": []}

    roots: List[Dict[str, Any]] = []
    for cid, node in nodes.items():
        parent_id = node["code"].get("parent_code_id")
        if parent_id is None:
            roots.append(node)
        else:
            pid = str(parent_id)
            parent = nodes.get(pid)
            if parent:
                parent["children"].append(node)
            else:
                # Orphaned node -> treat as root
                roots.append(node)
    return roots


def categorize_by_function(
    codes: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize codes by their frame function and build trees within each category."""
    categories = {
        "TOPIC": [],
        "BENEFIT_ATTRIBUTION": [],
        "RISK_ATTRIBUTION": [],
        "BENEFIT_EVALUATION": [],
        "RISK_EVALUATION": [],
        "TREATMENT": [],
    }

    # Group codes by function
    for c in codes:
        func = c.get("function", "OTHER")
        if func in categories:
            categories[func].append(c)
        else:
            # Put unknown functions in a default category
            if "OTHER" not in categories:
                categories["OTHER"] = []
            categories["OTHER"].append(c)

    # Build trees within each category
    categorized_trees = {}
    for func, func_codes in categories.items():
        if func_codes:  # Only include categories that have codes
            categorized_trees[func] = build_tree(func_codes)

    return categorized_trees


def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_node(node: Dict[str, Any]) -> str:
    c = node["code"]
    cid = escape_html(str(c.get("code_id") or c.get("id") or ""))
    name = escape_html(str(c.get("name") or ""))
    func = escape_html(str(c.get("function") or c.get("function", "")))
    desc = escape_html(str(c.get("description") or ""))
    merged = c.get("merged_candidates") or []
    is_root = c.get("parent_code_id") is None
    evidence_raw = c.get("evidence", {})
    evidence = normalize_evidence(evidence_raw)

    # Build HTML for merged codes (only surface for root codes)
    merged_html = ""
    if is_root and merged:
        items = []
        for m in merged:
            m_name = escape_html(str(m.get("name") or ""))
            m_id = escape_html(str(m.get("code_id") or ""))
            m_desc = escape_html(str(m.get("description") or ""))
            meta = f"[ID:{m_id}]" if m_id else ""
            items.append(f"<li><span class='merged-name'>{m_name}</span> <span class='merged-meta'>{meta}</span><div class='merged-desc'>{m_desc}</div></li>")
        merged_html = f"<div class='merged-codes'><strong>Merged codes:</strong><ul>{''.join(items)}</ul></div>"

    # Build HTML for evidence
    ev_html_parts = []
    for aid, quotes in evidence.items():
        aid_esc = escape_html(aid)
        ev_items = "\n".join(f"<li>{escape_html(q)}</li>" for q in quotes)
        ev_html_parts.append(
            f'<div class="evidence-block"><strong>Article {aid_esc}:</strong><ul>{ev_items}</ul></div>'
        )
    ev_html = "\n".join(ev_html_parts) if ev_html_parts else "<em>No evidence</em>"

    # children placeholder; will be filled by recursion in render_tree
    children_html = "".join(render_node(ch) for ch in node.get("children", []))

    html = f"""
    <li class="code-node">
      <div class="code-header">
        <span class="code-name">{name}</span>
        <span class="code-meta">[ID:{cid}]</span>
        <button class="toggle-btn" onclick="toggleDetails(this)">details</button>
      </div>
      <div class="code-details" style="display:none;">
        <div class="code-desc"><strong>Description:</strong> {desc}</div>
        {merged_html}
        <div class="code-evidence"><strong>Evidence:</strong>{ev_html}</div>
      </div>
      {('<ul class="children">' + children_html + '</ul>') if children_html else ''}
    </li>
    """
    return html


def render_tree(roots: List[Dict[str, Any]]) -> str:
    return "\n".join(render_node(r) for r in roots)


def render_categorized_trees(categorized_trees: Dict[str, List[Dict[str, Any]]]) -> str:
    """Render trees organized by frame function categories."""
    function_descriptions = {
        "TOPIC": "Central issue/topic discussed.",
        "BENEFIT_ATTRIBUTION": "Who is credited with producing positive outcomes.",
        "RISK_ATTRIBUTION": "Who is blamed for negative outcomes.",
        "BENEFIT_EVALUATION": "Positive judgments/benefits highlighted.",
        "RISK_EVALUATION": "Negative judgments/risks highlighted.",
        "TREATMENT": "Stance or recommendation (supportive vs critical).",
    }

    sections = []
    for func, trees in categorized_trees.items():
        func_esc = escape_html(func)
        desc = function_descriptions.get(func, "Other codes")
        desc_esc = escape_html(desc)
        tree_html = render_tree(trees)

        section = f"""
    <div class="function-section">
      <h2 class="function-header">
        <span class="function-name">{func_esc}</span>
        <span class="function-desc">({desc_esc})</span>
        <span class="function-count">{len([n for tree in trees for n in _count_nodes(tree)])} codes</span>
      </h2>
      <ul class="function-codes">
        {tree_html}
      </ul>
    </div>
        """
        sections.append(section)

    return "\n".join(sections)


def _count_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Helper to count all nodes in a tree (including children)."""
    result = [node]
    for child in node.get("children", []):
        result.extend(_count_nodes(child))
    return result


def generate_html(
    categorized_trees: Dict[str, List[Dict[str, Any]]], title: str = "Codebook"
) -> str:
    body = render_categorized_trees(categorized_trees)
    # Build a template without using f-string to avoid accidental brace
    # interpolation issues. We'll insert the title and timestamp safely.
    from datetime import datetime

    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{TITLE}</title>
<style>
  body { font-family: system-ui, -apple-system, Roboto, Arial, sans-serif; padding: 16px; }
  ul { list-style: none; padding-left: 20px; }
  .function-section { margin: 24px 0; border: 1px solid #ddd; border-radius: 8px; padding: 16px; }
  .function-header { margin: 0 0 16px 0; padding: 8px; background: #f5f5f5; border-radius: 4px; }
  .function-name { font-size: 1.2em; font-weight: 700; color: #333; }
  .function-desc { font-size: 0.9em; color: #666; margin-left: 8px; }
  .function-count { font-size: 0.8em; color: #999; margin-left: auto; }
  .function-codes { margin: 0; }
  .code-node { margin: 6px 0; padding: 6px; border-left: 2px solid #ccc; }
  .code-header { display:flex; gap:8px; align-items:center; }
  .code-name { font-weight:600; margin-right:6px; }
  .code-meta { color:#666; font-size:0.9em; }
  .toggle-btn { margin-left:auto; font-size:0.8em; }
  .code-details { background:#f9f9f9; padding:8px; margin-top:6px; border-radius:4px; }
  .evidence-block { margin:6px 0; }
  .merged-codes { margin:6px 0; padding:6px; background:#eef6ff; border-radius:4px; }
  .merged-name { font-weight:600; }
  .merged-meta { color:#666; font-size:0.85em; margin-left:4px; }
  .merged-desc { color:#444; font-size:0.9em; margin-left:4px; }
</style>
</head>
<body>
<h1>{TITLE}</h1>
<p>Generated: {TIMESTAMP}</p>
<div class="codebook-root">
{BODY}
</div>
<script>
function toggleDetails(btn){
  var details = btn.parentElement.nextElementSibling;
  if(!details) return;
  if(details.style.display === 'none') details.style.display = 'block';
  else details.style.display = 'none';
}
</script>
</body>
</html>"""

    html = (
        template.replace("{TITLE}", escape_html(title))
        .replace("{TIMESTAMP}", escape_html(datetime.now().isoformat()))
        .replace("{BODY}", body)
    )
    return html


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Visualize a codebook JSON as HTML")
    parser.add_argument("input", help="Path to codebook JSON file")
    parser.add_argument("output", nargs="?", help="Output HTML file (optional)")
    args = parser.parse_args(argv)

    inpath = args.input
    if not os.path.exists(inpath):
        print(f"Error: input file not found: {inpath}")
        sys.exit(2)

    outpath = args.output
    if not outpath:
        base = os.path.splitext(os.path.basename(inpath))[0]
        outpath = base + ".html"

    try:
        data = load_codebook(inpath)
        codes = data.get("codes") if isinstance(data, dict) else []
        if not codes:
            print(
                "Warning: no 'codes' found in JSON; attempting to treat file as list of codes"
            )
            if isinstance(data, list):
                codes = data

        # Normalize each code's evidence
        for c in codes:
            c["evidence"] = normalize_evidence(c.get("evidence", {}))

        categorized_trees = categorize_by_function(codes)
        html = generate_html(
            categorized_trees, title=f"Codebook: {os.path.basename(inpath)}"
        )
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✅ Wrote HTML visualization to {outpath}")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
