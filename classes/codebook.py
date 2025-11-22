from typing import Dict, Optional, List, Any
from classes.dataclasses import Code, Operation, OperationType, Function
from collections import defaultdict
from datetime import datetime


class Codebook:
    def __init__(self, vector_index=None):
        self.codes: Dict[int, Code] = {}
        self._next_id = 1
        self.index = vector_index

    # ------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------

    def _generate_id(self) -> int:
        """Return a new unique code ID."""
        new_id = self._next_id
        self._next_id += 1
        return new_id

    def _get(self, code_id: int) -> Optional[Code]:
        return self.codes.get(code_id)

    def add_code(self, code: Code) -> Code:
        """Add code to the codebook."""
        if code.code_id is None:
            code.code_id = self._generate_id()
        self.codes[code.code_id] = code
        return code

    def delete_code(self, code_id: int) -> bool:
        """Delete a code from the codebook."""
        if code_id in self.codes:
            del self.codes[code_id]
            # Sync with vector index
            if self.index:
                self.index.remove(code_id)
            return True
        return False

    # ------------------------------------------------------------
    # Core 6 Operation Handlers
    # ------------------------------------------------------------

    def apply_operation(self, op: Operation) -> None:
        """Apply one of the 6 supported operations to the codebook."""
        match op.operation_type:
            case OperationType.CREATE_CODE:
                self._op_create_code(op)

            case OperationType.MERGE_TO_EXISTING:
                self._op_merge_to_existing(op)

            case OperationType.MERGE_TO_CANDIDATE:
                self._op_merge_to_candidate(op)

            case OperationType.CREATE_PARENT_GROUP:
                self._op_create_parent_group(op)

            case OperationType.NO_ACTION:
                return  # do nothing

            case _:
                raise ValueError(f"Unsupported operation type: {op.operation_type}")

    # ------------------------------------------------------------
    # 1. CREATE_CODE  (Candidate code becomes new leaf)
    # ------------------------------------------------------------
    def _op_create_code(self, op: Operation) -> None:
        if not op.new_code_data:
            raise ValueError("CREATE_CODE requires new_code_data")

        # Create new Code and populate optional fields (description, embedding)
        new_code = Code(
            code_id=self._generate_id(),
            name=op.new_code_data["name"],
            function=op.new_code_data["function"],
            description=op.new_code_data.get("description", ""),
            evidence=defaultdict(list),
            embedding=op.new_code_data.get("embedding", None),
        )

        # Add evidence if provided
        if "evidence" in op.new_code_data:
            for article_id, quotes in op.new_code_data["evidence"].items():
                new_code.evidence[article_id].extend(quotes)

        self.add_code(new_code)

    # ------------------------------------------------------------
    # 2. MERGE_TO_EXISTING (candidate → existing)
    # ------------------------------------------------------------
    def _op_merge_to_existing(self, op: Operation) -> None:
        """Merge candidate evidence into an existing code. Candidate is not kept."""

        existing = self._get(int(op.target_code_id))
        if not existing:
            raise ValueError("MERGE_TO_EXISTING requires valid target_code_id")

        # Evidence passed from candidate via op.new_code_data["candidate"]
        candidate_info = op.new_code_data.get("candidate")
        if not candidate_info:
            print("⚠ MERGE_TO_EXISTING missing candidate data — nothing to merge")
            return

        candidate_evidence = candidate_info.get("evidence", {})

        # -------- SAFE MERGING --------
        if isinstance(candidate_evidence, dict):
            for aid, quotes in candidate_evidence.items():
                existing.evidence[int(aid)].extend(quotes)
        else:
            print("⚠ Malformed candidate evidence:", candidate_evidence)

        # Update name if decision agent suggested a better one
        new_name = op.new_code_data.get("name")
        if new_name:
            existing.name = new_name

        # Track merged-in candidate code metadata
        existing.merged_candidates.append(
            {
                "code_id": candidate_info.get("code_id"),
                "name": candidate_info.get("name"),
                "description": candidate_info.get("description"),
                "function": candidate_info.get("function"),
            }
        )

        existing.updated_at = datetime.utcnow()

    # ------------------------------------------------------------
    # 3. MERGE_TO_CANDIDATE (existing → candidate)
    # ------------------------------------------------------------
    def _op_merge_to_candidate(self, op: Operation) -> None:
        """Candidate becomes the new canonical label."""
        existing = self._get(int(op.source_code_id))

        # Determine candidate id; if none provided, generate a new one
        candidate_id = (
            int(op.target_code_id) if op.target_code_id is not None else self._generate_id()
        )

        # Handle candidate data from new_code_data (candidate is not in codebook yet)
        candidate = self._get(candidate_id)
        if not candidate and op.new_code_data and "candidate" in op.new_code_data:
            candidate_data = op.new_code_data["candidate"]
            candidate = Code(
                code_id=candidate_id,
                name=candidate_data["name"],
                function=Function(candidate_data["function"]),
                description=candidate_data.get("description", ""),
                evidence=defaultdict(list),
            )
            # Add evidence if provided
            candidate_evidence = candidate_data.get("evidence", {})
            if isinstance(candidate_evidence, list):
                article_id = getattr(op, "article_id", 1)
                for quote in candidate_evidence:
                    candidate.evidence[article_id].append(quote)
            elif isinstance(candidate_evidence, dict):
                for article_id, quotes in candidate_evidence.items():
                    candidate.evidence[int(article_id)].extend(quotes)
            self.add_code(candidate)

        if not existing or not candidate:
            raise ValueError("MERGE_TO_CANDIDATE requires valid IDs or candidate data")

        # Merge existing evidence into candidate
        for article_id, quotes in existing.evidence.items():
            candidate.evidence[article_id].extend(quotes)

        candidate.updated_at = datetime.utcnow()

        # Track merged-in codes
        candidate.merged_candidates.append(
            {
                "code_id": existing.code_id,
                "name": existing.name,
                "description": existing.description,
                "function": existing.function.value if hasattr(existing.function, "value") else str(existing.function),
            }
        )

        # Optionally update candidate name
        if op.new_code_data and op.new_code_data.get("name"):
            candidate.name = op.new_code_data["name"]

        # Delete old existing code
        self.delete_code(existing.code_id)

    # ------------------------------------------------------------
    # 4. CREATE_PARENT_GROUP (new parent created for both codes)
    # ------------------------------------------------------------
    def _op_create_parent_group(self, op: Operation) -> None:
        """Create a new parent node above an existing code and a new candidate code."""
        existing = self._get(int(op.target_code_id))
        if not existing:
            raise ValueError("CREATE_PARENT_GROUP requires valid target_code_id")

        # Candidates are never in the codebook yet - create the candidate first
        if not op.new_code_data or "candidate" not in op.new_code_data:
            raise ValueError("CREATE_PARENT_GROUP requires candidate data")

        candidate_data = op.new_code_data["candidate"]
        candidate = Code(
            code_id=self._generate_id(),
            name=candidate_data["name"],
            function=Function(candidate_data["function"]),
            description=candidate_data.get("description", ""),
            evidence=defaultdict(list),
        )

        # Add evidence if provided - ensure proper format
        if "evidence" in candidate_data and candidate_data["evidence"]:
            evidence_data = candidate_data["evidence"]
            if isinstance(evidence_data, dict):
                for article_id, quotes in evidence_data.items():
                    if isinstance(quotes, list):
                        candidate.evidence[int(article_id)].extend(quotes)
                    else:
                        candidate.evidence[int(article_id)].append(str(quotes))
        self.add_code(candidate)

        parent_name = op.new_code_data.get("parent_name")
        if not parent_name:
            # Generate a fallback parent name based on existing code's name
            fallback_name = f"Parent of {existing.name}"
            print(
                f"⚠ Warning: CREATE_PARENT_GROUP missing parent_name, using fallback: '{fallback_name}'"
            )
            parent_name = fallback_name

        # Create parent node
        new_parent = Code(
            code_id=self._generate_id(),
            name=parent_name,
            function=existing.function,  # Use function of existing code
            description=op.new_code_data.get("description", ""),
            evidence=defaultdict(list),
        )
        self.add_code(new_parent)

        # Re-parent both codes under the new parent
        existing.parent_code_id = new_parent.code_id
        candidate.parent_code_id = new_parent.code_id

    # ------------------------------------------------------------
    # 5. NO_OP (do nothing)
    # ------------------------------------------------------------
    # Handled at entrypoint

    def find_latest_code_id(self) -> Optional[int]:
        """
        Return the ID of the most recently created code.
        Assumes _generate_id() increments monotonically.
        """
        if not self.codes:
            return None
        return max(self.codes.keys())
