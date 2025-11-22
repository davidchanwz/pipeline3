
from typing import Optional
from clients.llm_client import LLMClient

class CodeDescriptionAgent:
    """Agent responsible for producing refined human-readable descriptions
    for codes when structural changes occur in the codebook.

    This implementation accepts an optional `LLMClient` which acts as an
    abstraction over any LLM provider. If a client is provided, the agent
    will attempt to use it; otherwise it will fall back to deterministic
    local implementations so the system remains runnable without network
    dependencies.
    """

    def __init__(
        self, llm: Optional[LLMClient] = None, system_prompt: Optional[str] = None
    ):
        self.llm = llm
        self.system_prompt = system_prompt

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Helper to call the injected LLM client safely.

        Returns the model response as string, or None on failure / if no client.
        """
        if not self.llm:
            return None
        try:
            # Keep kwargs minimal; concrete clients can accept more options.
            return self.llm.generate(prompt, system_prompt=self.system_prompt)
        except Exception:
            # Swallow exceptions and fall back to deterministic behavior
            return None

    def refine_merge_description(
        self, existing_desc: Optional[str], candidate_desc: Optional[str]
    ) -> str:
        """Compress and unify two descriptions into one (1–2 sentences).

        Attempts to use the injected LLM client; if unavailable or it fails,
        returns a deterministic 1–2 sentence merge of the inputs.
        """
        prompt = (
            "Refine these two code descriptions into a single 1–2 sentence, "
            "article-agnostic conceptual definition. Return only the definition.\n\n"
            f"Existing description:\n{existing_desc or ''}\n\n"
            f"Candidate description:\n{candidate_desc or ''}\n"
        )

        llm_response = self._call_llm(prompt)
        if llm_response:
            return llm_response.strip()

        # Deterministic fallback implementation
        parts = [p.strip() for p in (existing_desc or "").split(".") if p.strip()]
        parts += [p.strip() for p in (candidate_desc or "").split(".") if p.strip()]

        if not parts:
            return ""

        sentences = []
        for p in parts:
            if len(sentences) >= 2:
                break
            s = p.rstrip(".")
            sentences.append(s + ".")

        return " ".join(sentences)

    def refine_parent_description(
        self, child_desc_1: Optional[str], child_desc_2: Optional[str]
    ) -> str:
        """Produce an abstract parent description that generalizes both children.

        Tries LLM first; falls back to a concise deterministic summary.
        """
        prompt = (
            "Synthesize a short (1–2 sentence) parent-level description that "
            "abstracts and generalizes the following two child descriptions."
            " Return only the parent description.\n\n"
            f"Child A:\n{child_desc_1 or ''}\n\n"
            f"Child B:\n{child_desc_2 or ''}\n"
        )

        llm_response = self._call_llm(prompt)
        if llm_response:
            return llm_response.strip()

        a = (child_desc_1 or "").strip()
        b = (child_desc_2 or "").strip()

        if not a and not b:
            return ""

        def short(x: str) -> str:
            return (
                (x.split(".")[0][:120].strip() + ("..." if len(x) > 120 else ""))
                if x
                else x
            )

        if a and b:
            return f"A broader concept that generalizes: {short(a)} and {short(b)}."
        return a or b

    def refine_child_description(
        self, child_desc: Optional[str], parent_desc: Optional[str]
    ) -> str:
        """Refine a child description so it reads clearly as a subtype of the parent.

        Tries LLM first; deterministic fallback used on failure.
        """
        prompt = (
            "Rewrite the child description so it clearly reads as a specific "
            "subtype of the parent. Keep it brief (1 sentence). Return only the "
            "rewritten child description.\n\n"
            f"Parent description:\n{parent_desc or ''}\n\n"
            f"Child description:\n{child_desc or ''}\n"
        )

        llm_response = self._call_llm(prompt)
        if llm_response:
            return llm_response.strip()

        c = (child_desc or "").strip()
        p = (parent_desc or "").strip()

        if not c and not p:
            return ""

        if not c:
            return f"A specific subtype of {p.split('.')[0]}." if p else ""

        if not p:
            return c

        if "subtype" in c.lower() or "type of" in c.lower():
            return c

        parent_short = p.split(".")[0]
        child_short = c.split(".")[0]
        return f"{child_short}. (A subtype of {parent_short}.)"
