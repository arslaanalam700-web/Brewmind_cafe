"""Structured logger and tracer for multi-agent pipeline executions."""
import json
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from config import (
    BASE_DIR,
    COST_PER_MILLION_INPUT_TOKENS,
    COST_PER_MILLION_OUTPUT_TOKENS
)
from .schema import AgentTrace, ExecutionSession

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TRACE_LOG_FILE = LOGS_DIR / "agent_traces.jsonl"

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate USD cost for Gemini Flash token usage."""
    input_cost = (prompt_tokens / 1_000_000.0) * COST_PER_MILLION_INPUT_TOKENS
    output_cost = (completion_tokens / 1_000_000.0) * COST_PER_MILLION_OUTPUT_TOKENS
    return input_cost + output_cost

class Tracer:
    """Session tracer managing active session and persisting trace steps."""
    
    def __init__(self):
        self.active_session: Optional[ExecutionSession] = None
        self.session_history: List[ExecutionSession] = []

    def start_session(self, query: str, retrieval_strategy: str = "hybrid_rerank") -> ExecutionSession:
        """Start a new end-to-end multi-agent session."""
        session_id = str(uuid.uuid4())[:8]
        self.active_session = ExecutionSession(
            session_id=session_id,
            query=query,
            retrieval_strategy=retrieval_strategy
        )
        return self.active_session

    def log_trace(
        self,
        agent_name: str,
        step_type: str,
        input_text: str,
        output_text: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentTrace:
        """Log a completed agent step trace."""
        trace_id = str(uuid.uuid4())[:8]
        total_tokens = prompt_tokens + completion_tokens
        cost = calculate_cost(prompt_tokens, completion_tokens)
        
        trace = AgentTrace(
            trace_id=trace_id,
            agent_name=agent_name,
            step_type=step_type,
            input_text=str(input_text),
            output_text=str(output_text),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            status=status,
            metadata=metadata or {}
        )
        
        if self.active_session:
            self.active_session.traces.append(trace)
            self.active_session.total_latency_ms += latency_ms
            self.active_session.total_tokens += total_tokens
            self.active_session.total_cost_usd += cost
            
        # Append to jsonl file
        try:
            with open(TRACE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict()) + "\n")
        except Exception as e:
            print(f"Warning: Failed to write trace log: {e}")
            
        return trace

    def end_session(self, final_report: str = "", unverified_claims: Optional[List[Dict[str, Any]]] = None) -> Optional[ExecutionSession]:
        """Finalize the active session."""
        if not self.active_session:
            return None
        self.active_session.final_report = final_report
        self.active_session.unverified_claims = unverified_claims or []
        self.session_history.append(self.active_session)
        session = self.active_session
        self.active_session = None
        return session

global_tracer = Tracer()
