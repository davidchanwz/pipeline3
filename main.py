#!/usr/bin/env python3
"""
main.py
-------
Easy command interface for pipeline3 services.

Usage:
    python main.py explore [model] [articles]     # Run multi-model explorer
    python main.py cluster [model|codebook]       # Run hierarchical clustering for GPT/Gemini or specific codebook
    python main.py viz                           # Visualize cluster means
    python main.py all [articles]               # Run complete pipeline
"""

import sys
import subprocess
import os


def explore(model=None, articles="5"):
    """Run multi-model explorer."""
    cmd = ["python", "services/multi_model_explorer.py", "--articles", str(articles)]

    if model:
        if model == "gpt":
            cmd.extend(["--models", "gpt-4o-mini"])
        elif model == "gemini":
            cmd.extend(["--models", "gemini-2.0-flash-lite"])
        elif model == "both":
            cmd.extend(["--models", "gpt-4o-mini", "gemini-2.0-flash-lite"])
        else:
            cmd.extend(["--models", model])

    print(f"🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def cluster(model=None):
    """Run hierarchical clustering."""
    cmd = ["python", "utils/hierarchical_clustering.py"]
    if model:
        if model in ["gpt", "gemini"]:
            cmd.extend(["--model", model])
        else:
            # Treat as direct codebook path override
            cmd.extend(["--codebook", model])
    print(f"📊 Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def visualize(viz_type="cluster", arg=None, codebook_path=None):
    """Visualize cluster means or codebook."""
    if viz_type in ["cluster", "means", "c"]:
        # arg supports model shortcut ("gpt"/"gemini") or explicit CSV path
        if arg in ["gpt", "gemini"]:
            csv_path = f"cluster_means_{arg}.csv"
        elif arg:
            csv_path = arg
        else:
            csv_path = "cluster_means.csv"
        cmd = ["python", "scripts/visualize_cluster_means.py", csv_path]
        print(f"📈 Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    elif viz_type in ["codebook", "book", "cb"]:
        # arg doubles as codebook path when provided
        target = codebook_path or arg
        cmd = ["python", "scripts/visualize_codebook.py"]
        if target:
            cmd.append(target)
        print(f"📊 Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        print(f"❌ Unknown visualization type: {viz_type}")
        print("💡 Use: cluster, means, codebook, book")


def run_all(articles="5"):
    """Run complete pipeline: explore -> cluster -> visualize."""
    print("🔄 Running complete pipeline...")

    # Step 1: Multi-model exploration
    print("\n=== Step 1: Multi-model exploration ===")
    explore("both", articles)

    # Step 2: Hierarchical clustering (on GPT codebook by default)
    print("\n=== Step 2: Hierarchical clustering ===")
    cluster("gpt")

    # Step 3: Visualization
    print("\n=== Step 3: Visualization ===")
    visualize("cluster", "gpt")

    print("\n✅ Complete pipeline finished!")
    print("📁 Check: *_codebook.json, cluster_means_gpt.csv, cluster_means_gpt.html")


def show_help():
    """Show available commands."""
    print(
        """
Pipeline3 Commands:

🔍 EXPLORATION:
  python main.py explore                    # Both models, 5 articles
  python main.py explore gpt               # GPT only, 5 articles  
  python main.py explore gemini            # Gemini only, 5 articles
  python main.py explore both 10           # Both models, 10 articles
  python main.py explore gpt-4o-mini 3     # Specific model, 3 articles

📊 ANALYSIS:
  python main.py cluster                   # Run hierarchical clustering (default GPT)
  python main.py cluster gemini            # Cluster using Gemini codebook
  python main.py cluster /path/to/codebook # Cluster with explicit codebook path
  python main.py viz                       # Visualize cluster means (default: cluster_means.csv)
  python main.py viz cluster gpt           # Visualize cluster means for GPT (cluster_means_gpt.csv)
  python main.py viz cluster gemini        # Visualize cluster means for Gemini
  python main.py viz codebook              # Visualize codebook hierarchy
  python main.py viz codebook gpt_*.json   # Visualize specific codebook

🚀 SHORTCUTS:
  python main.py all                       # Complete pipeline (5 articles)
  python main.py all 10                    # Complete pipeline (10 articles)

💡 EXAMPLES:
  python main.py explore gemini 3          # Quick Gemini test with 3 articles
  python main.py all 15                    # Full analysis with 15 articles
"""
    )


def main():
    """Main command dispatcher."""
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    if command in ["explore", "exp", "e"]:
        model = sys.argv[2] if len(sys.argv) > 2 else "both"
        articles = sys.argv[3] if len(sys.argv) > 3 else "5"
        explore(model, articles)

    elif command in ["cluster", "clust", "c"]:
        model = sys.argv[2] if len(sys.argv) > 2 else None
        cluster(model)

    elif command in ["visualize", "viz", "v"]:
        viz_type = sys.argv[2] if len(sys.argv) > 2 else "cluster"
        arg = sys.argv[3] if len(sys.argv) > 3 else None
        visualize(viz_type, arg)

    elif command in ["all", "pipeline", "full", "a"]:
        articles = sys.argv[2] if len(sys.argv) > 2 else "5"
        run_all(articles)

    elif command in ["help", "h", "--help", "-h"]:
        show_help()

    else:
        print(f"❌ Unknown command: {command}")
        print("💡 Try: python main.py help")


if __name__ == "__main__":
    main()
