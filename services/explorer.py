"""
explorer.py
-----------
High-level orchestrator for automated hierarchical codebook construction.
"""

from typing import List, Dict, Any, Optional
from agents.candidate_coder import CandidateCoder
from utils.pipeline_logger import get_pipeline_logger


class Explorer:
    """
    Orchestrates the pipeline:

    Article → LLM → Candidate Codes
             ↓
        VectorIndex (top-k similar existing codes)
             ↓
        DecisionAgent (choose 1 of 6 operations)
             ↓
        CodebookOperator (mutate codebook)
             ↓
        VectorIndex (update embedding)
    """

    def __init__(
        self,
        llm,  # LLMClient
        strategy,  # Strategy
        codebook,  # Codebook
        operator,  # CodebookOperator
        index,  # VectorIndex
        decision_agent,  # DecisionAgent
        top_k=5,
        similarity_threshold=0.6,
        candidate_coder: Optional[CandidateCoder] = None,
    ):
        self.llm = llm
        self.strategy = strategy
        self.codebook = codebook
        self.operator = operator
        self.index = index
        self.decision_agent = decision_agent
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        # If a CandidateCoder is not provided, construct a default from
        # the provided llm and strategy so callers don't have to create it.
        if candidate_coder is None:
            self.candidate_coder = CandidateCoder(llm=self.llm, strategy=self.strategy)
        else:
            self.candidate_coder = candidate_coder

    # ------------------------------------------------------------
    # Public Entry Point
    # ------------------------------------------------------------
    def run(self, articles: List[Dict[str, Any]]) -> Any:
        """
        Processes a list of articles:
            [{"id": 1, "title": "...", "content": "..."}, ...]
        """

        for article in articles:
            print(f"\n=== PROCESSING ARTICLE {article['id']} ===")

            # 1. LLM extracts candidate codes
            candidates = self._extract_candidates(article)

            # 2. Integrate each candidate into the codebook
            for cand in candidates:
                self._handle_candidate(article_id=article["id"], candidate=cand)

        print("\n=== COMPLETED ALL ARTICLES ===")

        # Show logging summary
        logger = get_pipeline_logger()
        print(f"📊 Logged {logger.get_decision_count()} decisions")
        print(f"📁 Full log available at: {logger.get_log_file_path()}")

        return self.codebook

    # ------------------------------------------------------------
    # Step 1: Extract LLM Candidate Codes
    # ------------------------------------------------------------
    def _extract_candidates(self, article) -> List[Dict[str, Any]]:
        # Delegate candidate extraction/parsing to the CandidateCoder agent.
        return self.candidate_coder.extract_candidates(article)

    # ------------------------------------------------------------
    # Step 2: Process a single candidate
    # ------------------------------------------------------------
    def _handle_candidate(self, article_id: int, candidate: Dict[str, Any]):
        """
        Pipeline:
        1. Embed candidate
        2. Retrieve similar codes
        3. Decide operation
        4. Apply operation
        5. Update embeddings in index
        """

        # 1. Compute candidate embedding
        text_for_embedding = f"{candidate['name']}: {candidate['description']}"
        candidate_embedding = self.index.embed_text(text_for_embedding)
        candidate["embedding"] = candidate_embedding

        # 2. Retrieve top-k similar codes
        similar_codes = self.index.search(
            candidate_embedding,
            top_k=self.top_k,
            min_score=self.similarity_threshold,
            function_filter=candidate.get("function"),
        )

        # 3. Ask LLM which operation to perform
        operation = self.decision_agent.decide(candidate, similar_codes, article_id)

        # 4. Apply operation to the codebook
        self.operator.apply_and_refine(self.codebook, operation)

        # 5. Update embeddings for all codes that don't have them
        # (apply_and_refine may have created new codes without embeddings)
        for code in self.codebook.codes.values():
            if not hasattr(code, "embedding") or code.embedding is None:
                self.index.add_or_update(code)
