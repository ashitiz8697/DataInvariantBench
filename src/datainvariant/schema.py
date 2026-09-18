from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColumnSpec:
    name: str
    dtype: str
    description: str
    unit: Optional[str] = None
    missing_value: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ColumnSpec":
        return cls(**value)


@dataclass
class TableSpec:
    name: str
    description: str
    columns: List[ColumnSpec]
    rows: List[Dict[str, Any]]
    primary_key: Optional[List[str]] = None
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TableSpec":
        payload = dict(value)
        payload["columns"] = [ColumnSpec.from_dict(item) for item in value["columns"]]
        return cls(**payload)


@dataclass
class AnswerSpec:
    value: Any
    value_type: str
    unit: Optional[str] = None
    tolerance: float = 0.0
    rounding: Optional[int] = None
    rounding_mode: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "AnswerSpec":
        return cls(**value)


@dataclass
class BenchmarkCase:
    case_id: str
    pair_id: str
    family: str
    variant: str
    question: str
    tables: List[TableSpec]
    answer: AnswerSpec
    transformation_contract: Dict[str, Any]
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "BenchmarkCase":
        payload = dict(value)
        payload["tables"] = [TableSpec.from_dict(item) for item in value["tables"]]
        payload["answer"] = AnswerSpec.from_dict(value["answer"])
        return cls(**payload)

    def clone(self, case_id: str, variant: str) -> "BenchmarkCase":
        payload = self.to_dict()
        payload["case_id"] = case_id
        payload["variant"] = variant
        return BenchmarkCase.from_dict(payload)


@dataclass
class Prediction:
    case_id: str
    agent: str
    status: str
    value: Any = None
    sql: Optional[str] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Prediction":
        return cls(**value)
