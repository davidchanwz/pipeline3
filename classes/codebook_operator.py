from typing import Optional

from agents.code_description_agent import CodeDescriptionAgent
from classes.dataclasses import Operation, OperationType
from classes.codebook import Codebook


class CodebookOperator:
    """Orchestrates applying structural operations to a Codebook and then
    refining affected descriptions using a CodeDescriptionAgent.

    Usage:
        agent = CodeDescriptionAgent()
        operator = CodebookOperator(agent)
        operator.apply_and_refine(codebook, operation)
    """

    def __init__(self, agent: CodeDescriptionAgent):
        self.agent = agent

    def apply_and_refine(self, codebook: Codebook, op: Operation) -> None:
        """
        Apply the structural operation, then refine/update the affected descriptions.
        This version:
        - Supports merge-first logic
        - Applies new_name / new_description
        - Correctly tracks which code survives
        - Prevents ID misalignment in MERGE_TO_CANDIDATE
        """

        op_type = op.operation_type

        # ----------------------------------------------------------------------
        # CAPTURE ORIGINAL DESCRIPTIONS BEFORE STRUCTURAL CHANGES OCCUR
        # ----------------------------------------------------------------------
        existing_before = None
        candidate_before = None

        existing_id = op.target_code_id     # surviving code for MERGE_TO_EXISTING, parent for ATTACH
        old_existing_id = op.source_code_id # code that gets merged & deleted for MERGE_TO_CANDIDATE

        if existing_id is not None:
            ex_obj = codebook._get(existing_id)
            existing_before = ex_obj.description if ex_obj else None

        if old_existing_id is not None:
            old_obj = codebook._get(old_existing_id)
            candidate_before = old_obj.description if old_obj else None

        # Also track the candidate's own description (pre-creation)
        # Useful for merges where the candidate is the “new main” code.
        candidate_raw_desc = None
        if op.new_code_data and isinstance(op.new_code_data.get("candidate"), dict):
            candidate_raw_desc = op.new_code_data["candidate"].get("description")

        # ----------------------------------------------------------------------
        # APPLY STRUCTURAL OPERATION — CODEBOOK MAY DELETE/CREATE NODES HERE
        # ----------------------------------------------------------------------
        result = codebook.apply_operation(op)

        # NOTE: some apply_operation() impls return the ID of the new/updated code.
        # If yours doesn't, we detect the surviving code below.

        # ----------------------------------------------------------------------
        # HANDLE: MERGE_TO_EXISTING
        # ----------------------------------------------------------------------
        if op_type == OperationType.MERGE_TO_EXISTING:
            surviving = codebook._get(op.target_code_id)
            if not surviving:
                return

            # Apply LLM-updated name/description when provided
            if op.new_code_data.get("name"):
                surviving.name = op.new_code_data["name"]
            if op.new_code_data.get("description"):
                surviving.description = op.new_code_data["description"]
            else:
                # Otherwise call the LLM description merger
                merged = self.agent.refine_merge_description(
                    existing_before, candidate_raw_desc or candidate_before
                )
                if merged:
                    surviving.description = merged
            return

        # ----------------------------------------------------------------------
        # HANDLE: MERGE_TO_CANDIDATE
        # ----------------------------------------------------------------------
        if op_type == OperationType.MERGE_TO_CANDIDATE:
            # After operation: a new code exists representing the merged result.
            # The operator should return its ID; otherwise find the newest leaf.
            surviving_id = getattr(result, "new_code_id", None)

            if surviving_id is None:
                # Fallback: attempt to resolve by comparing old_existing_id deletion
                surviving_id = codebook.find_latest_code_id()

            surviving = codebook._get(surviving_id)
            if not surviving:
                return

            # Apply LLM "name" and "description" overrides
            if op.new_code_data.get("name"):
                surviving.name = op.new_code_data["name"]

            if op.new_code_data.get("description"):
                surviving.description = op.new_code_data["description"]
            else:
                # Merge both descriptions if LLM did not override directly
                merged = self.agent.refine_merge_description(existing_before, candidate_raw_desc)
                if merged:
                    surviving.description = merged
            return

        # ----------------------------------------------------------------------
        # CREATE_CODE & NO_ACTION — no refinement needed
        # ----------------------------------------------------------------------
        return
