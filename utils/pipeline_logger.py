import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from classes.dataclasses import Code, Operation


class PipelineLogger:
    """
    Dedicated logging for pipeline decision-making process.
    Captures candidate codes, similar codes, decisions, and operations.
    """

    def __init__(self, log_dir: str = "logs", model_name: Optional[str] = None):
        self.log_dir = log_dir
        self.model_name = model_name
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model-specific log file if model_name is provided
        if model_name:
            clean_model = (
                model_name.replace(".", "_").replace("/", "_").replace("-", "_")
            )
            self.log_file = os.path.join(
                log_dir, f"pipeline_{clean_model}_{self.session_id}.json"
            )
        else:
            self.log_file = os.path.join(log_dir, f"pipeline_{self.session_id}.json")

        self.decisions = []

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

    def log_decision(
        self,
        article_id: int,
        candidate_name: str,
        candidate_description: str,
        candidate_evidence: Any,
        candidate_function: str,
        similar_codes: List[Code],
        llm_decision: Dict[str, Any],
        final_operation: Operation,
        reasoning: str = "",
    ):
        """Log a complete decision cycle with all relevant information."""

        # Format similar codes for logging
        similar_codes_data = []
        for code in similar_codes:
            similar_codes_data.append(
                {
                    "id": str(code.code_id),
                    "name": code.name,
                    "description": code.description,
                    "evidence": self._format_evidence(code.evidence),
                    "function": getattr(code.function, "value", str(code.function)),
                    "parent_id": getattr(code, "parent_code_id", None),
                }
            )

        # Create decision log entry
        decision_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "article_id": article_id,
            "candidate": {
                "name": candidate_name,
                "description": candidate_description,
                "evidence": self._format_evidence(candidate_evidence),
                "function": candidate_function,
            },
            "similar_codes": similar_codes_data,
            "llm_decision": llm_decision,
            "final_operation": {
                "type": final_operation.operation_type.value,
                "target_code_id": getattr(final_operation, "target_code_id", None),
                "source_code_id": getattr(final_operation, "source_code_id", None),
                "confidence": final_operation.confidence,
                "reasoning": final_operation.reasoning,
                "new_code_data": getattr(final_operation, "new_code_data", None),
            },
            "additional_reasoning": reasoning,
        }

        self.decisions.append(decision_entry)
        self._write_to_file()

        # Also print a summary to console
        self._print_decision_summary(decision_entry)

    def _format_evidence(self, evidence: Any) -> Any:
        """Format evidence for clean logging."""
        if isinstance(evidence, dict):
            # Convert defaultdict or regular dict to regular dict for JSON serialization
            return dict(evidence)
        return evidence

    def _write_to_file(self):
        """Write the current decisions to the log file."""
        try:
            with open(self.log_file, "w") as f:
                json.dump(
                    {
                        "session_id": self.session_id,
                        "model_name": self.model_name,
                        "decisions": self.decisions,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        except Exception as e:
            print(f"⚠ Warning: Failed to write to log file: {e}")

    def _print_decision_summary(self, entry: Dict[str, Any]):
        """Print a clean summary of the decision to console."""
        candidate = entry["candidate"]
        similar_count = len(entry["similar_codes"])
        operation = entry["final_operation"]

        model_info = f" - {entry['model_name']}" if entry.get("model_name") else ""
        print(f"\n📋 DECISION LOG (Article {entry['article_id']}){model_info}")
        print(f"   Candidate: '{candidate['name']}' ({candidate['function']})")
        print(
            f"   Description: {candidate['description'][:80]}{'...' if len(candidate['description']) > 80 else ''}"
        )
        print(f"   Evidence: {self._truncate_evidence(candidate['evidence'])}")

        if similar_count > 0:
            print(f"   Similar Codes ({similar_count}):")
            for i, sim in enumerate(entry["similar_codes"][:3]):  # Show first 3
                print(f"     {i+1}. ID:{sim['id']} '{sim['name']}' ({sim['function']})")
            if similar_count > 3:
                print(f"     ... and {similar_count - 3} more")
        else:
            print("   Similar Codes: None found")

        print(
            f"   🎯 Decision: {operation['type']} (confidence: {operation['confidence']:.2f})"
        )
        print(f"   💭 Reasoning: {operation['reasoning']}")

        if operation.get("new_code_data"):
            new_data = operation["new_code_data"]
            if "parent_name" in new_data:
                print(f"   👨‍👩‍👧‍👦 Parent Name: {new_data['parent_name']}")

        print("   " + "─" * 60)

    def _truncate_evidence(self, evidence: Any, max_length: int = 100) -> str:
        """Truncate evidence for display."""
        evidence_str = str(evidence)
        if len(evidence_str) <= max_length:
            return evidence_str
        return evidence_str[:max_length] + "..."

    def get_log_file_path(self) -> str:
        """Get the path to the current log file."""
        return self.log_file

    def get_decision_count(self) -> int:
        """Get the number of decisions logged."""
        return len(self.decisions)


# Model-specific logger instances
_logger_instances: Dict[str, PipelineLogger] = {}


def get_pipeline_logger(model_name: Optional[str] = None) -> PipelineLogger:
    """Get or create a pipeline logger instance, optionally model-specific."""
    global _logger_instances

    # Use model_name as key, or 'default' for backward compatibility
    key = model_name or "default"

    if key not in _logger_instances:
        _logger_instances[key] = PipelineLogger(model_name=model_name)

    return _logger_instances[key]


def log_decision(
    article_id: int,
    candidate_name: str,
    candidate_description: str,
    candidate_evidence: Any,
    candidate_function: str,
    similar_codes: List[Code],
    llm_decision: Dict[str, Any],
    final_operation: Operation,
    reasoning: str = "",
    model_name: Optional[str] = None,
):
    """Convenience function to log a decision using a model-specific logger."""
    logger = get_pipeline_logger(model_name=model_name)
    logger.log_decision(
        article_id=article_id,
        candidate_name=candidate_name,
        candidate_description=candidate_description,
        candidate_evidence=candidate_evidence,
        candidate_function=candidate_function,
        similar_codes=similar_codes,
        llm_decision=llm_decision,
        final_operation=final_operation,
        reasoning=reasoning,
    )
