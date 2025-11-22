#!/usr/bin/env python3
"""
test_explorer_10_articles.py
----------------------------
Test script to run the Explorer pipeline on articles 1-10 from articles.json
and generate a comprehensive codebook. This tests the full pipeline with multiple articles.
"""

import json
import os
import sys
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, try to load manually
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all necessary classes
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


def load_articles() -> list:
    """Load articles from data/articles.json"""
    with open("data/articles.json", "r") as f:
        return json.load(f)


def setup_pipeline():
    """Set up all components of the Explorer pipeline"""

    # 1. Create core components
    strategy = Strategy()
    codebook = Codebook()

    # 2. Create OpenAI LLM client
    llm = OpenAIClient(model="gpt-4o-mini")

    # 3. Create embedding service (use OpenAI for embeddings)
    embed_service = EmbeddingService(openai_model="text-embedding-3-small")

    # 4. Create vector index
    vector_index = VectorIndex(codebook, embed_service)

    # 5. Create description agent and operator
    description_agent = CodeDescriptionAgent(llm=llm)
    codebook_operator = CodebookOperator(description_agent)

    # 6. Create decision agent
    decision_agent = DecisionAgent(llm)

    # 7. Create candidate coder
    candidate_coder = CandidateCoder(llm, strategy)

    # 8. Create explorer
    explorer = Explorer(
        llm=llm,
        strategy=strategy,
        codebook=codebook,
        operator=codebook_operator,
        index=vector_index,
        decision_agent=decision_agent,
        candidate_coder=candidate_coder,
        top_k=5,  # Increase for better similarity matching across more articles
        similarity_threshold=0.5,
    )

    return explorer, codebook


def save_codebook(codebook: Codebook, filename: str = "test10codebook.json"):
    """Save the codebook to JSON format"""

    # Convert codebook to serializable format
    codebook_data = {
        "metadata": {
            "created_by": "test_explorer_10_articles.py",
            "source_articles": "Articles 1-10",
            "framework": "Entman Framing Theory",
            "total_codes": len(codebook.codes),
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
            "embedding": code.embedding,
            "created_at": code.created_at.isoformat() if code.created_at else None,
            "updated_at": code.updated_at.isoformat() if code.updated_at else None,
            "parent_code_id": code.parent_code_id,
        }
        codebook_data["codes"].append(code_dict)

    with open(filename, "w") as f:
        json.dump(codebook_data, f, indent=2)

    print(f"✅ Codebook saved to {filename}")
    print(f"   Contains {len(codebook_data['codes'])} codes")


def print_codebook_summary(codebook: Codebook):
    """Print a summary of the generated codebook"""

    print(f"\n📊 CODEBOOK SUMMARY")
    print(f"   Total codes: {len(codebook.codes)}")

    # Group by function
    function_counts = {}
    parent_codes = []
    child_codes = []

    for code in codebook.codes.values():
        func = (
            code.function.value
            if hasattr(code.function, "value")
            else str(code.function)
        )
        function_counts[func] = function_counts.get(func, 0) + 1

        if code.parent_code_id is None:
            parent_codes.append(code)
        else:
            child_codes.append(code)

    print(f"   Function distribution:")
    for func, count in function_counts.items():
        print(f"     - {func}: {count} codes")

    print(
        f"   Hierarchy: {len(parent_codes)} parent codes, {len(child_codes)} child codes"
    )

    print(f"\n📋 CODE DETAILS:")
    for code in codebook.codes.values():
        hierarchy_mark = "└─" if code.parent_code_id else "📁"
        func = (
            code.function.value
            if hasattr(code.function, "value")
            else str(code.function)
        )

        # Handle evidence in both dict and list formats
        if isinstance(code.evidence, dict):
            evidence_count = sum(len(quotes) for quotes in code.evidence.values())
        elif isinstance(code.evidence, list):
            evidence_count = len(code.evidence)
        else:
            evidence_count = 0

        print(
            f"   {hierarchy_mark} ID:{code.code_id} '{code.name}' ({func}) - {evidence_count} evidence quotes"
        )


def main():
    """Run the test on articles 1-10 and generate comprehensive codebook"""
    print("🚀 Starting Explorer pipeline test on articles 1-10...")

    try:
        # 1. Load articles
        articles = load_articles()
        test_articles = articles[:10]  # Get first 10 articles
        print(f"📄 Loaded {len(test_articles)} articles for processing")

        for i, article in enumerate(test_articles, 1):
            title = article.get("title", "Untitled")[:50]
            print(f"   Article {i}: {title}{'...' if len(title) >= 50 else ''}")

        # 2. Set up pipeline
        print("\n⚙️  Setting up pipeline components...")
        explorer, codebook = setup_pipeline()

        # 3. Run explorer on articles 1-10
        print(f"\n🔄 Running Explorer on {len(test_articles)} articles...")
        result_codebook = explorer.run(test_articles)

        # 4. Print detailed results
        print_codebook_summary(result_codebook)

        # 5. Save codebook
        save_codebook(result_codebook)

        print(
            f"\n✅ Test completed successfully! Processed {len(test_articles)} articles."
        )
        print("📁 Comprehensive codebook saved as test10codebook.json")

    except Exception as e:
        print(f"❌ Error during pipeline test: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
