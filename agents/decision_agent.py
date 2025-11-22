import json
from typing import List, Dict, Any, Optional
from clients.llm_client import LLMClient
from classes.dataclasses import Operation, OperationType, Code, Function
from services.strategy import Strategy
from utils.pipeline_logger import log_decision


class DecisionAgent:
    """
    The AI decision-maker for hierarchical codebook editing.
    Given:
    - a candidate code (dict or Code-like dict)
    - a list of similar existing Code objects
    Returns:
    A structured Operation object ready for CodebookOperator.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm
        # Extract model name from LLM client if available
        self.model_name = getattr(llm, "model", None) or getattr(
            llm, "model_name", None
        )

    # ------------------------------------------------------------ #
    # Public decision entrypoint                                    #
    # ------------------------------------------------------------ #

    def decide(
        self, candidate: dict, similar_codes: List[Code], article_id: int = None
    ) -> Operation:
        """
        High-level decision function.
        Returns an Operation object to be passed to CodebookOperator.
        """
        prompt = self._build_prompt(candidate, similar_codes)
        raw_response = self.llm.generate(prompt=prompt)
        decision = self._parse_llm_json(raw_response)

        final_operation = self._convert_to_operation(
            decision, candidate, similar_codes, article_id
        )

        # Log the complete decision process
        log_decision(
            article_id=article_id or 0,
            candidate_name=candidate.get("name", ""),
            candidate_description=candidate.get("description", ""),
            candidate_evidence=candidate.get("evidence", {}),
            candidate_function=candidate.get("function", ""),
            similar_codes=similar_codes,
            llm_decision=decision,
            final_operation=final_operation,
            reasoning=(
                f"LLM response: {raw_response[:200]}..."
                if len(raw_response) > 200
                else raw_response
            ),
            model_name=self.model_name,
        )

        return final_operation

    # ------------------------------------------------------------ #
    # Prompt builder                                                #
    # ------------------------------------------------------------ #

    def _build_prompt(self, candidate: dict, similar_codes: List[Code]) -> str:
        """
        Build the consolidated multi-code decision prompt for the LLM.
        """
        # Use the Strategy prompts to build our decision prompt. Strategy
        # provides both a decision context and a per-existing-code decision
        # prompt; we combine them here so the LLM sees the researcher's task
        # guidance plus the specific existing-code vs candidate comparison.
        strat = Strategy()

        # Base context (general principles for the decision)
        context = strat.get_decision_context()

        # Candidate info (kept simple)
        candidate_block = (
            f"CANDIDATE CODE\n--------------\n"
            f"Name: {candidate.get('name')}\n"
            f"Function: {candidate.get('function')}\n"
            f"Description: {candidate.get('description')}\n"
            f"Evidence: {candidate.get('evidence')}\n"
        )

        # If we have similar codes, create a prompt for the first one using
        # Strategy.get_decision_prompt. For additional similar codes, append
        # brief summaries to aid the model.
        similar_prompt_parts: List[str] = []
        brief_list: List[str] = []

        if similar_codes:
            # Use the first similar code as the primary comparison
            primary = similar_codes[0]
            existing_name = primary.name
            existing_function = (
                primary.function.value
                if hasattr(primary.function, "value")
                else str(primary.function)
            )
            existing_description = getattr(primary, "description", "")
            existing_evidence = getattr(primary, "evidence", "")
            existing_created = getattr(primary, "created_at", "")

            # Strategy's prompt expects strings; convert evidence and dates
            existing_evidence_str = str(existing_evidence)
            existing_created_str = str(existing_created)

            primary_prompt = strat.get_decision_prompt(
                existing_code_id=str(primary.code_id),
                existing_code_name=existing_name,
                existing_code_function=existing_function,
                existing_code_description=str(existing_description),
                existing_code_evidence=existing_evidence_str,
                existing_code_created=existing_created_str,
                candidate_code_name=candidate.get("name"),
                candidate_code_function=candidate.get("function"),
                candidate_code_description=str(candidate.get("description")),
                candidate_code_evidence=str(candidate.get("evidence")),
            )
            similar_prompt_parts.append(primary_prompt)

            # Append short summaries for any additional similar codes
            for c in similar_codes[1:]:
                brief_list.append(
                    f"ID:{c.code_id} Name:{c.name} Function:{getattr(c.function, 'value', c.function)}"
                )

        full_prompt = context + "\n\n" + candidate_block + "\n\n"

        if similar_prompt_parts:
            full_prompt += "\n\n".join(similar_prompt_parts)

        if brief_list:
            full_prompt += "\n\nADDITIONAL RELATED CODES:\n" + "\n".join(brief_list)

        # Always include JSON response format instructions
        # When there are no similar codes, the Strategy decision prompt isn't used,
        # so we need to add the format instructions manually
        if not similar_prompt_parts:
            full_prompt += (
                "\n\n"
                + """
------------------------------------
RESPONSE FORMAT (STRICT JSON ONLY)
------------------------------------
{
  "operation": "CREATE_NEW",
  "confidence": 0.1-1.0,
  "reasoning": "Short justification"
}

Since there are no existing similar codes, use CREATE_NEW operation to add this as a new code.
"""
            )

        return full_prompt

    # ------------------------------------------------------------ #
    # JSON Parsing                                                  #
    # ------------------------------------------------------------ #

    def _parse_llm_json(self, raw: str) -> Dict[str, Any]:
        """
        Attempts strict JSON parsing.
        If parsing fails, returns fallback NO_ACTION.
        """
        # Clean up the response - remove markdown code fences if present
        cleaned_raw = raw.strip()

        if cleaned_raw.startswith("```json"):
            cleaned_raw = cleaned_raw[7:]  # Remove ```json
        if cleaned_raw.startswith("```"):
            cleaned_raw = cleaned_raw[3:]  # Remove generic ```
        if cleaned_raw.endswith("```"):
            cleaned_raw = cleaned_raw[:-3]  # Remove trailing ```

        cleaned_raw = cleaned_raw.strip()

        try:
            parsed = json.loads(cleaned_raw)

            # Handle case where LLM returns an array instead of a single object
            if isinstance(parsed, list):
                if len(parsed) > 0 and isinstance(parsed[0], dict):
                    return parsed[0]  # Take the first decision
                else:
                    raise ValueError("Empty array or non-dict elements")
            elif isinstance(parsed, dict):
                return parsed
            else:
                raise ValueError("Parsed JSON is neither dict nor array")

        except Exception as e:
            print(f"⚠ Warning: JSON parse failed ({e}), using NO_ACTION fallback")
            print(
                f"🔍 Debug: Cleaned decision response (first 200 chars): {repr(cleaned_raw[:200])}"
            )
            return {
                "operation": "NO_ACTION",
                "target_ids": [],
                "confidence": 0.0,
                "reasoning": "Failed to parse LLM JSON",
                "new_name": None,
                "parent_name": None,
            }

    # ------------------------------------------------------------ #
    # Convert LLM decision → Operation object                       #
    # ------------------------------------------------------------ #
    def _convert_to_operation(
        self,
        decision: Dict[str, Any],
        candidate: dict,
        similar_codes: List[Code],
        article_id: int,
    ) -> Operation:
        """
        Converts the LLM decision dict into a fully structured Operation object.
        Ensures evidence structure is consistently normalized.
        """

        # -------------------------------
        # Normalize evidence consistently
        # -------------------------------
        def normalize_evidence(raw):
            """Always convert to dict {article_id: [quotes]}"""
            if raw is None:
                return {}

            if isinstance(raw, list):
                return {article_id: raw}

            if isinstance(raw, dict):
                # Clean dict keys into ints
                return {int(k): list(v) for k, v in raw.items()}

            return {}

        operation_str = decision.get("operation")
        if operation_str == "CREATE_NEW":
            operation_str = "CREATE_CODE"
        elif operation_str == "NO_OP":
            operation_str = "NO_ACTION"

        try:
            op_type = OperationType(operation_str)
        except ValueError:
            print(f"⚠ Unknown operation '{operation_str}' from LLM — using NO_ACTION")
            op_type = OperationType.NO_ACTION
        target_ids = decision.get("target_ids", [])

        # Normalize candidate evidence BEFORE passing into any operation
        candidate_evidence = normalize_evidence(candidate.get("evidence"))

        # ----------------------------------------------------------
        # CREATE_CODE
        # ----------------------------------------------------------
        if op_type == OperationType.CREATE_CODE:
            return Operation(
                operation_type=OperationType.CREATE_CODE,
                new_code_data={
                    "name": candidate["name"],
                    "function": candidate["function"],
                    "description": candidate.get("description", ""),
                    "evidence": candidate_evidence,
                },
                confidence=decision.get("confidence", 0.0),
                reasoning=decision.get("reasoning", "")
            )

        # ----------------------------------------------------------
        # MERGE_TO_EXISTING
        # ----------------------------------------------------------
        if op_type == OperationType.MERGE_TO_EXISTING:
            if not target_ids:
                print("⚠ MERGE_TO_EXISTING missing target_id → fallback to CREATE_CODE")
                return Operation(
                    operation_type=OperationType.CREATE_CODE,
                    new_code_data={
                        "name": candidate["name"],
                        "function": candidate["function"],
                        "description": candidate.get("description", ""),
                        "evidence": candidate_evidence,
                    }
                )

            return Operation(
                operation_type=OperationType.MERGE_TO_EXISTING,
                target_code_id=int(target_ids[0]),
                new_code_data={
                    "name": decision.get("new_name", None),
                    "description": decision.get("new_description"),
                    "candidate": {
                        **candidate,
                        "evidence": candidate_evidence,
                    },
                },
                confidence=decision.get("confidence", 0.0),
                reasoning=decision.get("reasoning", "")
            )

        # ----------------------------------------------------------
        # MERGE_TO_CANDIDATE
        # ----------------------------------------------------------
        if op_type == OperationType.MERGE_TO_CANDIDATE:
            if not target_ids:
                print("⚠ MERGE_TO_CANDIDATE missing target_id → fallback to CREATE_CODE")
                return Operation(
                    operation_type=OperationType.CREATE_CODE,
                    new_code_data={
                        "name": candidate["name"],
                        "function": candidate["function"],
                        "description": candidate.get("description", ""),
                        "evidence": candidate_evidence,
                    }
                )

            return Operation(
                operation_type=OperationType.MERGE_TO_CANDIDATE,
                source_code_id=int(target_ids[0]),
                target_code_id=candidate.get("code_id"),
                new_code_data={
                    "name": decision.get("new_name", candidate["name"]),
                    "description": decision.get("new_description"),
                    "candidate": {
                        **candidate,
                        "evidence": candidate_evidence,
                    },
                },
                confidence=decision.get("confidence", 0.0),
                reasoning=decision.get("reasoning", "")
            )

        # ----------------------------------------------------------
        return Operation(
            operation_type=OperationType.NO_ACTION,
            confidence=decision.get("confidence", 0.0),
            reasoning=decision.get("reasoning", "")
        )
