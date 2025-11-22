from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime


# Entman's 4 frame functions
class Function(Enum):
    TOPIC = "TOPIC"
    BENEFIT_ATTRIBUTION = "BENEFIT_ATTRIBUTION"
    RISK_ATTRIBUTION = "RISK_ATTRIBUTION"
    BENEFIT_EVALUATION = "BENEFIT_EVALUATION"
    RISK_EVALUATION = "RISK_EVALUATION"
    TREATMENT = "TREATMENT"


@dataclass
class Code:
    code_id: int
    name: str
    function: Function
    description: str = ""
    merged_candidates: list = field(default_factory=list)  # Track merged-in codes
    evidence: defaultdict[int, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )  # article id : list of quotes
    embedding: Optional[List[float]] = None  # Semantic embedding vector
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
    parent_code_id: int = None

    def __post_init__(self):
        """Validate code properties after initialization."""
        # Ensure name is never empty
        if not self.name or str(self.name).strip() == "":
            self.name = f"Code {self.code_id}" if self.code_id else "Unnamed Code"


@dataclass
class ResearchFramework:
    """Defines a research framework for qualitative analysis."""

    name: str
    description: str
    functions: List[Function]

    def validate(self) -> bool:
        """Validate that the framework is complete and valid."""
        if not self.name or not self.description:
            return False
        if not self.functions:
            return False
        # Ensure no duplicate functions
        if len(set(self.functions)) != len(self.functions):
            return False
        return True

    @classmethod
    def create_entman_framework(cls) -> "ResearchFramework":
        """Create the default Entman framing framework."""
        return cls(
            name="Singapore Higher Education Frames",
            description=(
                "Frame elements adapted for higher education coverage in Singapore: "
                "topic/actors, attribution of benefits or risks, moral evaluation of benefits or risks, "
                "and treatment recommendations (positive or negative stance decided per code)."
            ),
            functions=[
                Function.TOPIC,
                Function.BENEFIT_ATTRIBUTION,
                Function.RISK_ATTRIBUTION,
                Function.BENEFIT_EVALUATION,
                Function.RISK_EVALUATION,
                Function.TREATMENT,
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert framework to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "functions": [func.value for func in self.functions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchFramework":
        """Create framework from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            functions=[Function(func) for func in data["functions"]],
        )


class OperationType(Enum):
    CREATE_CODE = "CREATE_CODE"
    DELETE_CODE = "DELETE_CODE"
    MERGE_TO_EXISTING = "MERGE_TO_EXISTING"
    MERGE_TO_CANDIDATE = "MERGE_TO_CANDIDATE"
    CREATE_PARENT_GROUP = "CREATE_PARENT_GROUP"
    NO_ACTION = "NO_ACTION"


@dataclass
class Operation:
    """Represents an operation to be performed on the codebook."""

    operation_type: OperationType
    target_code_id: Optional[int] = None
    source_code_id: Optional[int] = None  # For merge operations
    new_code_data: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    reasoning: str = ""
