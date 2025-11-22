#!/usr/bin/env python3
"""
multi_model_explorer.py
-----------------------
Service that runs the Explorer pipeline concurrently on multiple LLM models
(OpenAI, Gemini) and saves separate codebooks for comparison.

Usage:
    python services/multi_model_explorer.py [--articles N] [--models MODEL1 MODEL2 ...]
"""

import json
import os
import sys
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

# Import pipeline components
from classes.dataclasses import Code, Function, ResearchFramework
from classes.codebook import Codebook
from classes.vector_index import VectorIndex
from classes.codebook_operator import CodebookOperator
from services.strategy import Strategy
from services.embedder import EmbeddingService
from services.explorer import Explorer
from agents.decision_agent import DecisionAgent
from agents.candidate_coder import CandidateCoder
from agents.code_description_agent import CodeDescriptionAgent
from clients.openai_client import OpenAIClient
from clients.gemini_client import GeminiClient
from utils.pipeline_logger import get_pipeline_logger


class MultiModelExplorer:
    """Service for running Explorer pipeline on multiple models concurrently."""

    def __init__(self, models: List[str], num_articles: int = 10):
        self.models = models
        self.num_articles = num_articles
        self.results = {}

    def load_articles(self) -> List[Dict[str, Any]]:
        """Load articles from data/articles.json"""
        with open("data/articles.json", "r") as f:
            articles = json.load(f)
        return articles[: self.num_articles]

    def create_pipeline(self, model_name: str) -> tuple[Explorer, Codebook]:
        """Create a complete Explorer pipeline for a specific model."""

        # 1. Create core components
        strategy = Strategy()
        codebook = Codebook()

        # 2. Create LLM client based on model type
        if model_name.startswith("gemini"):
            llm = GeminiClient(model=model_name)
        else:
            # Default to OpenAI for gpt- models and others
            llm = OpenAIClient(model=model_name)

        # 3. Create embedding service (use consistent embedding model across all runs)
        embed_service = EmbeddingService(openai_model="text-embedding-3-small")

        # 4. Create vector index
        vector_index = VectorIndex(codebook, embed_service)
        codebook.index = vector_index

        # 5. Create agents
        description_agent = CodeDescriptionAgent(llm=llm)
        codebook_operator = CodebookOperator(description_agent)
        decision_agent = DecisionAgent(llm)
        candidate_coder = CandidateCoder(llm, strategy)

        # 6. Create explorer with model-specific configuration
        explorer = Explorer(
            llm=llm,
            strategy=strategy,
            codebook=codebook,
            operator=codebook_operator,
            index=vector_index,
            decision_agent=decision_agent,
            candidate_coder=candidate_coder,
            top_k=5,
            similarity_threshold=0.5,
        )

        return explorer, codebook

    def run_single_model(
        self, model_name: str, articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run the Explorer pipeline on a single model."""

        print(f"🚀 Starting pipeline for model: {model_name}")
        start_time = datetime.now()

        try:
            # Create pipeline
            explorer, codebook = self.create_pipeline(model_name)

            # Run pipeline
            result_codebook = explorer.run(articles)

            # Calculate metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Count codes by function
            function_counts = {}
            parent_count = 0
            child_count = 0
            total_evidence = 0

            for code in result_codebook.codes.values():
                func = (
                    code.function.value
                    if hasattr(code.function, "value")
                    else str(code.function)
                )
                function_counts[func] = function_counts.get(func, 0) + 1

                if code.parent_code_id is None:
                    parent_count += 1
                else:
                    child_count += 1

                # Count evidence quotes
                if hasattr(code, "evidence") and code.evidence:
                    for quotes in code.evidence.values():
                        total_evidence += len(quotes)

            result = {
                "model": model_name,
                "success": True,
                "duration_seconds": duration,
                "total_codes": len(result_codebook.codes),
                "parent_codes": parent_count,
                "child_codes": child_count,
                "total_evidence_quotes": total_evidence,
                "function_distribution": function_counts,
                "codebook": result_codebook,
                "completed_at": end_time.isoformat(),
            }

            print(
                f"✅ Completed {model_name}: {len(result_codebook.codes)} codes in {duration:.1f}s"
            )
            return result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            result = {
                "model": model_name,
                "success": False,
                "error": str(e),
                "duration_seconds": duration,
                "completed_at": end_time.isoformat(),
            }

            print(f"❌ Failed {model_name}: {e}")
            return result

    def save_codebook(
        self, model_name: str, codebook: Codebook, metadata: Dict[str, Any]
    ):
        """Save codebook to JSON file named after the model."""

        # Clean model name for filename (replace dots and slashes)
        clean_name = model_name.replace(".", "_").replace("/", "_").replace("-", "_")
        filename = f"{clean_name}_codebook.json"

        # Convert codebook to serializable format
        codebook_data = {
            "metadata": {
                "model": model_name,
                "created_by": "multi_model_explorer.py",
                "articles_processed": self.num_articles,
                "framework": "Entman Framing Theory",
                "total_codes": metadata.get("total_codes", 0),
                "duration_seconds": metadata.get("duration_seconds", 0),
                "function_distribution": metadata.get("function_distribution", {}),
                "completed_at": metadata.get("completed_at", ""),
            },
            "codes": [],
        }

        for code_id, code in codebook.codes.items():
            code_dict = {
                "code_id": code.code_id,
                "name": code.name,
                "function": (
                    code.function.value
                    if hasattr(code.function, "value")
                    else str(code.function)
                ),
                "description": code.description,
                "evidence": dict(code.evidence),  # Convert defaultdict to dict
                "merged_candidates": code.merged_candidates,
                "embedding": code.embedding,
                "created_at": code.created_at.isoformat() if code.created_at else None,
                "updated_at": code.updated_at.isoformat() if code.updated_at else None,
                "parent_code_id": code.parent_code_id,
            }
            codebook_data["codes"].append(code_dict)

        with open(filename, "w") as f:
            json.dump(codebook_data, f, indent=2)

        print(f"💾 Saved {filename} ({len(codebook.codes)} codes)")
        return filename

    def run_concurrent(self) -> Dict[str, Any]:
        """Run Explorer pipeline on all models concurrently."""

        print(
            f"🔄 Running Explorer on {len(self.models)} models with {self.num_articles} articles"
        )
        print(f"📋 Models: {', '.join(self.models)}")

        # Load articles once
        articles = self.load_articles()
        print(f"📄 Loaded {len(articles)} articles")

        # Use ThreadPoolExecutor for concurrent execution
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            # Submit all model runs
            future_to_model = {
                executor.submit(self.run_single_model, model, articles): model
                for model in self.models
            }

            # Collect results as they complete
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    results[model] = result

                    # Save codebook if successful
                    if result["success"]:
                        self.save_codebook(model, result["codebook"], result)

                except Exception as e:
                    print(f"❌ Exception in {model}: {e}")
                    results[model] = {"model": model, "success": False, "error": str(e)}

        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary comparison of all model results."""

        print("\n" + "=" * 80)
        print("📊 MULTI-MODEL COMPARISON SUMMARY")
        print("=" * 80)

        successful_runs = [r for r in results.values() if r["success"]]
        failed_runs = [r for r in results.values() if not r["success"]]

        print(f"✅ Successful runs: {len(successful_runs)}/{len(results)}")
        if failed_runs:
            print(f"❌ Failed runs: {len(failed_runs)}")
            for run in failed_runs:
                print(f"   - {run['model']}: {run.get('error', 'Unknown error')}")

        if successful_runs:
            print(f"\n📈 CODE GENERATION COMPARISON:")
            print(
                f"{'Model':<20} {'Total':<8} {'Parents':<8} {'Children':<8} {'Evidence':<8} {'Duration':<8}"
            )
            print("-" * 70)

            for run in sorted(
                successful_runs, key=lambda x: x["total_codes"], reverse=True
            ):
                print(
                    f"{run['model']:<20} "
                    f"{run['total_codes']:<8} "
                    f"{run['parent_codes']:<8} "
                    f"{run['child_codes']:<8} "
                    f"{run['total_evidence_quotes']:<8} "
                    f"{run['duration_seconds']:<8.1f}s"
                )

            # Function distribution comparison
            print(f"\n🏷️  FUNCTION DISTRIBUTION:")
            all_functions = set()
            for run in successful_runs:
                all_functions.update(run["function_distribution"].keys())

            print(
                f"{'Model':<20} "
                + " ".join(f"{func[:6]:<7}" for func in sorted(all_functions))
            )
            print("-" * (20 + len(all_functions) * 8))

            for run in successful_runs:
                dist = run["function_distribution"]
                counts = " ".join(
                    f"{dist.get(func, 0):<7}" for func in sorted(all_functions)
                )
                print(f"{run['model']:<20} {counts}")

        print("\n🏁 Multi-model comparison complete!")

        # Show model-specific log file info
        print("\n📁 Model-specific log files:")
        total_decisions = 0
        for model in self.models:
            logger = get_pipeline_logger(model_name=model)
            decisions = logger.get_decision_count()
            total_decisions += decisions
            print(f"   {model}: {logger.get_log_file_path()} ({decisions} decisions)")

        print(f"📊 Total decisions across all models: {total_decisions}")


def main():
    """Main entry point with command line argument parsing."""

    parser = argparse.ArgumentParser(
        description="Run Explorer pipeline on multiple LLM models (OpenAI, Gemini)"
    )
    parser.add_argument(
        "--articles",
        type=int,
        default=10,
        help="Number of articles to process (default: 10)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini", "gemini-2.0-flash-lite"],
        help="Models to test (default: gpt-4o-mini gemini-2.0-flash-lite)",
    )

    args = parser.parse_args()

    models = args.models

    # Create and run multi-model explorer
    explorer = MultiModelExplorer(models=models, num_articles=args.articles)
    results = explorer.run_concurrent()
    explorer.print_summary(results)


if __name__ == "__main__":
    main()
