from typing import List, Dict, Any
import json

from clients.llm_client import LLMClient
from services.strategy import Strategy


class CandidateCoder:
    """Encapsulate candidate-code extraction from an article using the
    project's Strategy prompt and an LLM client.

    Behavior:
    - Build the coding prompt via Strategy.get_coding_prompt
    - Call the provided LLM client's generate(...) method
    - Attempt to parse the result using llm.safe_json_parse if available
      otherwise fall back to json.loads
    - On any parse or generation error, return an empty list and print a
      short warning so the caller can continue processing.
    """

    def __init__(self, llm: LLMClient, strategy: Strategy):
        self.llm = llm
        self.strategy = strategy

    def build_prompt(self, article: Dict[str, Any]) -> str:
        """Return the coding prompt for a given article dict.

        Expects article to contain keys like 'title' and 'content'.
        """
        return self.strategy.get_coding_prompt(
            article_title=article.get("title"),
            article_content=article.get("content"),
        )

    def extract_candidates(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run the LLM to extract candidate codes from the article.

        Returns a list of candidate dicts, or an empty list on failure.
        """
        prompt = self.build_prompt(article)

        try:
            raw = self.llm.generate(prompt=prompt)
        except Exception as exc:  # network/LLM error
            print(f"⚠ Warning: LLM generate() failed: {exc}")
            return []

        # Clean up the response - remove markdown code fences if present
        cleaned_raw = raw.strip()
        if cleaned_raw.startswith("```json"):
            cleaned_raw = cleaned_raw[7:]  # Remove ```json
        if cleaned_raw.startswith("```"):
            cleaned_raw = cleaned_raw[3:]  # Remove generic ```
        if cleaned_raw.endswith("```"):
            cleaned_raw = cleaned_raw[:-3]  # Remove trailing ```
        cleaned_raw = cleaned_raw.strip()

        # Prefer provider's safe JSON parser if available
        try:
            if hasattr(self.llm, "safe_json_parse") and callable(
                getattr(self.llm, "safe_json_parse")
            ):
                return self.llm.safe_json_parse(cleaned_raw)

            # otherwise try strict JSON parse
            return json.loads(cleaned_raw)
        except Exception as exc:
            print(f"⚠ Warning: failed to parse LLM response as JSON: {exc}")
            print(
                f"🔍 Debug: Cleaned response (first 200 chars): {repr(cleaned_raw[:200])}"
            )
            return []
