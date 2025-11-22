# Pipeline 3

Research pipeline for hierarchical codebook construction, clustering, and visualization.

## Setup (uv)
1. Install deps with [uv](https://docs.astral.sh/uv/):
   ```
   uv sync
   ```
2. Set API keys (e.g., in `.env`):
   ```
   OPENAI_API_KEY=your_key
   GEMINI_API_KEY=your_key
   ```
3. Activate the virtual environment if needed:
   ```
   source .venv/bin/activate
   ```

## Core commands (main.py)
- Explore (LLM extraction + merge): `python main.py explore [model] [articles]`
  - Examples: `python main.py explore both 5`, `python main.py explore gpt 3`, `python main.py explore gemini`
- Cluster (build cluster means CSV): `python main.py cluster [gpt|gemini|/path/to/codebook.json]`
- Visualize cluster means: `python main.py viz cluster [gpt|gemini|/path/to/cluster_means.csv]`
- Visualize codebook tree: `python main.py viz codebook [/path/to/codebook.json]`
- Full pipeline (explore → cluster → visualize): `python main.py all [articles]`

## Outputs
- Codebooks: `*_codebook.json` plus HTML views.
- Cluster means: `cluster_means_<model>.csv` and `.html` (e.g., `cluster_means_gpt.csv`).

## Defaults & notes
- Model shortcuts:
  - `gpt` → `gpt_4o_mini_codebook.json`
  - `gemini` → `gemini_2_0_flash_lite_codebook.json`
- Clustering infers article count from evidence; override with `--total-articles` when running `utils/hierarchical_clustering.py` directly.
