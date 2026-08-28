"""5-Agent Orchestration Pipeline with SequentialAgent and LoopAgent refinement."""
import time
from typing import Dict, Any, Optional, List, Callable
from google import genai

from config import (
    GEMINI_API_KEY,
    DEFAULT_MODEL,
    MAX_VERIFIER_LOOPS
)
from src.observability.tracer import global_tracer, ExecutionSession
from .tools import set_active_retrieval_strategy, get_active_retrieval_strategy
from .planner import run_planner
from .retriever_agent import run_retriever_agent
from .writer import run_writer
from .verifier import run_verifier
from .editor import run_editor

class LiteratureReviewPipeline:
    """Orchestrates the 5-agent literature review generation workflow."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        retrieval_strategy: str = "hybrid_rerank",
        max_verifier_loops: int = MAX_VERIFIER_LOOPS
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model
        self.retrieval_strategy = retrieval_strategy
        self.max_verifier_loops = max_verifier_loops
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        
        # Set active strategy in tools
        set_active_retrieval_strategy(retrieval_strategy)

    def set_retrieval_strategy(self, strategy: str):
        """Update active retrieval strategy."""
        self.retrieval_strategy = strategy
        set_active_retrieval_strategy(strategy)

    def run(
        self,
        research_question: str,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute the full 5-agent pipeline.
        
        Args:
            research_question: High-level research question.
            progress_callback: Optional callback(stage_name, payload) for live UI updates.
            
        Returns:
            Dictionary with final report, trace session, evidence chunks, and verification stats.
        """
        if not self.client:
            self.client = genai.Client(api_key=self.api_key or GEMINI_API_KEY)
            
        session = global_tracer.start_session(
            query=research_question,
            retrieval_strategy=self.retrieval_strategy
        )
        
        # -------------------------------------------------------------
        # STEP 1: Planner Agent
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback("planner_start", {"question": research_question})
            
        plan_data = run_planner(
            research_question=research_question,
            client=self.client,
            model=self.model
        )
        
        if progress_callback:
            progress_callback("planner_done", {"plan": plan_data})
            
        # -------------------------------------------------------------
        # STEP 2: Retriever Agent
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback("retriever_start", {"strategy": self.retrieval_strategy})
            
        evidence_data = run_retriever_agent(plan_data=plan_data)
        
        if progress_callback:
            progress_callback("retriever_done", {"evidence": evidence_data})
            
        # -------------------------------------------------------------
        # STEP 3 & 4: Writer <-> Verifier Refinement Loop (LoopAgent)
        # -------------------------------------------------------------
        draft_text = ""
        verification_result = {}
        unverified_claims = []
        feedback = None
        
        for iteration in range(1, self.max_verifier_loops + 1):
            if progress_callback:
                progress_callback("writer_start", {"iteration": iteration, "has_feedback": bool(feedback)})
                
            writer_output = run_writer(
                research_question=research_question,
                evidence_data=evidence_data,
                verifier_feedback=feedback,
                iteration=iteration,
                client=self.client,
                model=self.model
            )
            draft_text = writer_output["draft"]
            
            if progress_callback:
                progress_callback("writer_done", {"iteration": iteration, "draft": draft_text})
                progress_callback("verifier_start", {"iteration": iteration})
                
            verification_result = run_verifier(
                draft_text=draft_text,
                iteration=iteration,
                client=self.client,
                model=self.model
            )
            
            if progress_callback:
                progress_callback("verifier_done", {"iteration": iteration, "verification": verification_result})
                
            # Check stopping condition
            if verification_result.get("overall_approved", False):
                print(f"[OK] Verifier approved draft at iteration {iteration} (Faithfulness: {verification_result.get('faithfulness_score', 1.0):.2f})")
                break
            else:
                feedback = verification_result.get("feedback_to_writer", "")
                print(f"[!] Verifier requested revisions (Iter {iteration}): {feedback[:100]}...")
                
        # If still unverified claims remain after max loops, capture them
        assessments = verification_result.get("claims_assessment", [])
        unverified_claims = [a for a in assessments if a.get("verdict") == "UNSUPPORTED"]
        
        # -------------------------------------------------------------
        # STEP 5: Editor Agent
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback("editor_start", {})
            
        editor_output = run_editor(
            research_question=research_question,
            verified_draft=draft_text,
            unverified_claims=unverified_claims,
            client=self.client,
            model=self.model
        )
        final_report = editor_output["final_report"]
        
        if progress_callback:
            progress_callback("editor_done", {"final_report": final_report})
            
        # Finalize trace session
        completed_session = global_tracer.end_session(
            final_report=final_report,
            unverified_claims=unverified_claims
        )
        
        return {
            "research_question": research_question,
            "retrieval_strategy": self.retrieval_strategy,
            "plan": plan_data,
            "evidence": evidence_data,
            "final_report": final_report,
            "cited_chunk_ids": editor_output.get("cited_chunk_ids", []),
            "unverified_claims": unverified_claims,
            "verification_history": verification_result,
            "session_summary": completed_session.to_dict() if completed_session else {}
        }
