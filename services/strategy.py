import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes.dataclasses import ResearchFramework, Function


class Strategy:
    """Simple strategic layer for MVP - just provides framework and basic prompt."""

    def __init__(self):
        self.framework = ResearchFramework.create_entman_framework()
        self._reference_codebook = None

    def get_framework(self) -> ResearchFramework:
        """Get the research framework."""
        return self.framework

    def _load_reference_codebook(self) -> str:
        """Load and format the reference codebook as an example, truncating embeddings."""
        if self._reference_codebook is not None:
            return self._reference_codebook

        try:
            # Get the path to the reference codebook
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            codebook_path = os.path.join(script_dir, "data", "reference_codebook_NOMORE.json")

            if not os.path.exists(codebook_path):
                return ""

            with open(codebook_path, "r") as f:
                data = json.load(f)

            # Format for display with truncated embeddings and proper descriptions
            formatted_codes = []
            for code in data.get("codes", [])[
                :3
            ]:  # Show only first 3 codes as examples
                # Generate example descriptions if missing (for better examples)
                code_name = code.get("name", "")
                existing_desc = code.get("description", "").strip()

                # Provide meaningful example descriptions if the reference is empty
                example_descriptions = {
                    "Historic Achievement": "A significant milestone or unprecedented accomplishment that marks an important moment in institutional or organizational history.",
                    "Commitment to Quality Education": "Dedication to providing excellent educational standards and maintaining high academic performance across institutional programs.",
                    "Research Excellence": "Superior performance in research activities, innovation, and scholarly contributions that demonstrate institutional capabilities.",
                }

                display_description = (
                    existing_desc
                    if existing_desc
                    else example_descriptions.get(
                        code_name,
                        "Article-agnostic conceptual definition of what this code represents.",
                    )
                )

                formatted_codes.append(
                    f"""    {{{{
        "name": "{code_name}",
        "function": "{code.get('function', '')}",
        "description": "{display_description}",
        "evidence": {json.dumps(code.get('evidence', {}))},
        "explanation": "Brief explanation of how the evidence supports this code"
    }}}}"""
                )

            self._reference_codebook = "[\n" + ",\n".join(formatted_codes) + "\n]"
            return self._reference_codebook

        except Exception as e:
            print(f"⚠ Warning: Could not load reference codebook: {e}")
            return ""

    def get_coding_prompt(
        self, article_title: str = None, article_content: str = None
    ) -> str:
        """Return the Entman coding prompt, optionally formatted with article data.

        The prompt is constructed inline so no separate helper or stored
        attribute is required.
        """
        reference_example = self._load_reference_codebook()
        example_section = (
            f"""

EXAMPLE OUTPUT (from previous analysis):
{reference_example}

Follow this same format and quality standard.
"""
            if reference_example
            else ""
        )

        # Use conditional content based on whether we have article data
        if article_title and article_content:
            article_section = f"""Article Title: {article_title}
Article Content: {article_content}"""
        else:
            article_section = """Article Title: {article_title}
Article Content: {article_content}"""

        prompt = f"""You are analyzing higher-education media coverage (Singapore) using these frame elements:

1. TOPIC - What aspect of higher education is at issue (e.g., admissions, rankings, student well-being).
2. BENEFIT_ATTRIBUTION - Who is credited with producing positive outcomes (e.g., government policy, universities, industry).
3. RISK_ATTRIBUTION - Who is blamed for problems or negative outcomes (e.g., government, universities, industry, others).
4. BENEFIT_EVALUATION - Positive judgments or benefits highlighted (e.g., economic/innovation gains, improved access, stronger research/skills).
5. RISK_EVALUATION - Negative judgments or risks highlighted (e.g., stress/mental health, inequality/elitism, over-emphasis on rankings).
6. TREATMENT - Treatment recommendation or stance (LLM decides if the leaning is supportive/positive or critical/negative).

Your task:
Extract a set of qualitative codes from the article.  
For each code, generate:
- a clear, concise **name**
- the correct **function** (one of the 4 above)
- **evidence quotes** supporting the code
- a **REQUIRED brief conceptual description (1–2 sentences)** summarizing the true meaning of the code  

CRITICAL: The description field is MANDATORY and must:
→ Be article-agnostic, generalizable, and suitable for embedding
→ NOT simply restate the evidence or be empty
→ Define the underlying concept the code represents
→ Be 1-2 complete sentences explaining what this code means conceptually

DO NOT leave the description field empty.{example_section}

{article_section}

RESPOND IN STRICT JSON FORMAT:

[
{{
    "name": "Code name",
    "function": "TOPIC | BENEFIT_ATTRIBUTION | RISK_ATTRIBUTION | BENEFIT_EVALUATION | RISK_EVALUATION | TREATMENT", 
    "description": "REQUIRED: 1-2 sentence conceptual definition explaining what this code represents (NOT a summary of evidence)",
    "evidence": ["quote 1", "quote 2"],
    "explanation": "Brief explanation of how the evidence supports the code"
}}
]

Remember: Every code MUST have a meaningful description that defines the concept, not just empty text.
"""

        return prompt  # _get_default_prompt removed — prompt is now inlined in get_coding_prompt

    def get_framework_context(self) -> str:
        """Get the framework context for setting agent context."""
        framework = self.framework

        # Build function descriptions dynamically from framework
        function_descriptions = []
        for func in framework.functions:
            if func == Function.TOPIC:
                function_descriptions.append(
                    "- TOPIC: Central issue/topic discussed."
                )
            elif func == Function.BENEFIT_ATTRIBUTION:
                function_descriptions.append(
                    "- BENEFIT_ATTRIBUTION: Who is credited with producing positive outcomes."
                )
            elif func == Function.RISK_ATTRIBUTION:
                function_descriptions.append(
                    "- RISK_ATTRIBUTION: Who is blamed for negative outcomes or problems."
                )
            elif func == Function.BENEFIT_EVALUATION:
                function_descriptions.append(
                    "- BENEFIT_EVALUATION: Positive judgments/benefits highlighted."
                )
            elif func == Function.RISK_EVALUATION:
                function_descriptions.append(
                    "- RISK_EVALUATION: Negative judgments/risks highlighted."
                )
            elif func == Function.TREATMENT:
                function_descriptions.append(
                    "- TREATMENT: Stance or recommendation; model infers positive/supportive vs negative/critical."
                )

        return f"""You are analyzing this article using {framework.name} with these functions:
{chr(10).join(function_descriptions)}

Research Framework: {framework.name}
Description: {framework.description}"""

    def get_response_schema(self) -> dict:
        """Get the JSON schema for structured responses."""
        # Build enum values dynamically from framework functions
        function_enum = [func.value for func in self.framework.functions]

        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "function": {"type": "string", "enum": function_enum},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "explanation": {"type": "string"},
                },
                "required": ["name", "function", "evidence", "explanation"],
            },
        }

    def get_decision_prompt(
        self,
        existing_code_id: str,
        existing_code_name: str,
        existing_code_function: str,
        existing_code_description: str,
        existing_code_evidence: str,
        existing_code_created: str,
        candidate_code_name: str,
        candidate_code_function: str,
        candidate_code_description: str,
        candidate_code_evidence: str,
    ) -> str:
        """Get the prompt for LLM-based operation decisions using the merge-first strategy."""
        return f"""
    You are an expert qualitative researcher maintaining a hierarchical codebook.

    Your job is to decide how a NEW candidate code should modify the existing codebook
    using one of the allowed operations.

    ------------------------------------
    EXISTING CODE
    ------------------------------------
    ID: {existing_code_id}
    Name: {existing_code_name}
    Function: {existing_code_function}
    Description: {existing_code_description}
    Evidence: {existing_code_evidence}
    Created: {existing_code_created}

    ------------------------------------
    CANDIDATE CODE
    ------------------------------------
    Name: {candidate_code_name}
    Function: {candidate_code_function}
    Description: {candidate_code_description}
    Evidence: {candidate_code_evidence}

    ------------------------------------
    ALLOWED OPERATIONS
    ------------------------------------

    1. MERGE_TO_EXISTING  
    - Candidate and existing describe the SAME underlying concept or the candidate is a subtype that cleanly fits under the existing concept.
    - The existing code remains the main label.
    - Candidate evidence is added to existing.
    - You MAY propose a new_name AND new_description if the merged concept 
        needs clearer, broader, or more accurate representation.

    2. MERGE_TO_CANDIDATE  
    - Candidate label/description better represents the combined concept (including when it is the cleaner parent label for both).
    - Existing evidence moves under the candidate.
    - You MUST propose a new_name or new_description if needed to accurately 
        capture the merged meaning.

    3. CREATE_NEW  
    - Candidate expresses a meaning that does NOT reasonably fit under the 
        existing code and cannot be merged through name/description revision.

    4. NO_OP  
    - Candidate contributes no new meaning and should be ignored.

    ------------------------------------
    MERGE-FIRST PRINCIPLE (CRITICAL)
    ------------------------------------
    Your first priorities are **descriptive efficiency**, **hierarchical consistency**, and **avoiding code fragmentation**.

    Therefore:
    - If the concepts overlap even moderately OR the candidate is a subtype of the existing concept → choose a MERGE operation.
    - If the candidate can be merged by simply updating a name or description → MERGE.
    - If unsure between MERGE and CREATE_NEW → ALWAYS choose MERGE.
    - CREATE_NEW must be a last resort.

    ------------------------------------
    HIERARCHY DISCIPLINE (CRITICAL)
    ------------------------------------
    - Merge similar codes AND codes that are subtypes of the same conceptual parent, provided a concise parent label (≤5 words) can represent them.
    - If a merge would force a tangled, overly long parent name (>5 words) or blur concepts, do NOT merge; prefer CREATE_NEW.
    - When merging, propose a succinct parent-style "new_name" and a "new_description" that expresses the shared underlying concept.

    ------------------------------------
    RESPONSE FORMAT (STRICT JSON)
    ------------------------------------
    {{
        "operation": "MERGE_TO_EXISTING | MERGE_TO_CANDIDATE | CREATE_NEW | NO_OP",
        "confidence": 0.1-1.0,
        "reasoning": "Short justification",
        "target_ids": ["Use the ID from the existing code above"],
        "new_name": "Only for MERGE operations — optional but recommended",
        "new_description": "Only for MERGE operations — recommended when integrating meanings"
    }}

    IMPORTANT:
    - MERGE_TO_EXISTING and MERGE_TO_CANDIDATE MUST include target_ids containing the 
    existing code's ID.
    - new_name and new_description MUST be included when merging if the current label 
    does not fully represent the merged concept.
    """

    def get_decision_schema(self) -> dict:
        """JSON schema for merge-first hierarchical codebook strategy."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "MERGE_TO_EXISTING",
                        "MERGE_TO_CANDIDATE",
                        "CREATE_NEW",
                        "NO_OP",
                    ],
                },
                "confidence": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                "reasoning": {"type": "string"},

                # Used ONLY for merge operations, but optional in JSON
                "new_name": {
                    "type": "string",
                    "description": (
                        "Used for MERGE operations when renaming improves clarity or "
                        "better reflects the merged concept."
                    ),
                },
                "new_description": {
                    "type": "string",
                    "description": (
                        "Used for MERGE operations when the combined meaning of the "
                        "existing and candidate codes requires a refined description."
                    ),
                },

                "target_ids": {
                    "type": "array",
                    "description": "Required for merge operations",
                    "items": {"type": "string"},
                },
            },
            "required": ["operation", "confidence", "reasoning"],
        }

    def get_decision_context(self) -> str:
        """Context for the merge-first hierarchical codebook editing strategy."""
        return """
    You are an expert in qualitative coding, ontology construction, and conceptual consolidation.

    FRAME FUNCTION GLOSSARY (Singapore Higher Education)
    - TOPIC: Central issue/topic (e.g., admissions, rankings, student well-being).
    - BENEFIT_ATTRIBUTION: Who is credited with producing positive outcomes (e.g., government policy, universities, industry).
    - RISK_ATTRIBUTION: Who is blamed for negative outcomes or problems (e.g., government, universities, industry, others).
    - BENEFIT_EVALUATION: Positive judgments/benefits highlighted (e.g., economic/innovation gains, improved access, stronger research/skills).
    - RISK_EVALUATION: Negative judgments/risks highlighted (e.g., stress/mental health, inequality/elitism, over-emphasis on rankings).
    - TREATMENT: Stance or recommendation; infer whether the article is supportive/expansive (positive) or critical/constraining (negative).

    Your goal is to maintain a hierarchical codebook that is:
    - compact,
    - theoretically meaningful,
    - non-redundant,
    - and interpretably structured.

    GENERAL PRINCIPLES
    ------------------
    1. **MERGE-FIRST PHILOSOPHY**
    - If two codes describe overlapping or partially overlapping meanings,
        prefer merging rather than creating new codes.
    - Small differences in wording are NOT reasons to create separate codes.
    - If the candidate could fit under an existing concept after renaming or 
        rewriting the description, you MUST choose a merge operation.

    2. **HIERARCHICAL FIT**
    - Merge similar codes AND codes that are subtype/sibling variants of the same underlying concept when a concise parent label (≤5 words) can describe the combined concept.
    - If the only possible merged label would be convoluted (>5 words) or muddled, keep concepts separate and lean toward CREATE_NEW.

    3. **DESCRIPTIVE EFFICIENCY**
    - Codebook should avoid fragmentation.
    - Broader concepts should accumulate evidence.
    - Do NOT create unnecessary leaf codes unless the candidate meaning cannot
        be reasonably merged into any existing concept.

    4. **REFINING MERGED CODES**
    - When merging, you may propose a new code name for clarity or generality, especially to name the concise parent concept.
    - When merging, you may refine or rewrite the description to integrate both meanings.

    5. **CREATE_NEW AS LAST RESORT**
    - CREATE_NEW should be used only when no reasonable merge is possible, 
        even after renaming or description revision.

    6. **NO_OP RARELY USED**
    - Only when the candidate adds no new meaning AND merging is not appropriate.

    Your overall responsibility:
    Maintain a clean, compact, semantically coherent, and hierarchically meaningful codebook that maximizes informational value and minimizes redundancy.
    """
