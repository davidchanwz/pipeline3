#!/usr/bin/env python3
"""
temp_test_explorer.py
--------------------
Temporary script to test the Explorer pipeline on article 1 from articles.json
and generate a reference codebook. This tests the full pipeline end-to-end.
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
        top_k=3,  # Small for testing
        similarity_threshold=0.5,
    )

    return explorer, codebook


def save_reference_codebook(
    codebook: Codebook, filename: str = "data/reference_codebook_v2.json"
):
    """Save the codebook to JSON format"""

    # Convert codebook to serializable format
    codebook_data = {
        "metadata": {
            "created_by": "temp_test_explorer.py",
            "source_article": "Article 1 - NUS Global Ranking",
            "framework": "Entman Framing Theory",
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

    # Ensure data directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(codebook_data, f, indent=2)

    print(f"✅ Reference codebook saved to {filename}")
    print(f"   Contains {len(codebook_data['codes'])} codes")


def main():
    """Run the test and generate reference codebook"""
    print("🚀 Starting Explorer pipeline test...")

    try:
        # 1. Load articles
        articles = load_articles()
        article_1 = articles[0]  # Get first article
        print(f"📄 Loaded article 1: {article_1.get('title', 'Untitled')[:60]}...")

        # 2. Set up pipeline
        print("⚙️  Setting up pipeline components...")
        explorer, codebook = setup_pipeline()

        # 3. Run explorer on article 1 only
        print("🔄 Running Explorer on article 1...")
        result_codebook = explorer.run([article_1])

        # 4. Print results
        print(f"\n📊 Pipeline completed successfully!")
        print(f"   Generated {len(result_codebook.codes)} codes")

        for code_id, code in result_codebook.codes.items():
            print(f"   - Code {code_id}: {code.name} ({code.function})")

        # 5. Save reference codebook
        save_reference_codebook(result_codebook)

        print("\n✅ Test completed successfully! The pipeline works end-to-end.")
        print("📁 Reference codebook saved as data/reference_codebook_v2.json")

    except Exception as e:
        print(f"❌ Error during pipeline test: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
