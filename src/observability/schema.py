"""Data schemas for agent traces and execution sessions."""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class AgentTrace:
    """Represents a single agent execution step."""
    trace_id: str
    agent_name: str
    step_type: str  # "planning", "retrieval", "writing", "verification", "editing"
    input_text: str
    output_text: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    status: str = "success"  # "success", "warning", "error"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ExecutionSession:
    """Represents the end-to-end execution of a research query."""
    session_id: str
    query: str
    retrieval_strategy: str
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    end_time: Optional[str] = None
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    traces: List[AgentTrace] = field(default_factory=list)
    final_report: str = ""
    unverified_claims: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
