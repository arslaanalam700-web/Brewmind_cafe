"""Evaluation harness package for benchmarking retrieval strategies and generation faithfulness."""
from .metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    hit_rate_at_k,
    compute_retrieval_metrics
)
from .llm_judge import evaluate_faithfulness, evaluate_relevance
from .run_eval import run_benchmark_eval
from .generate_report import generate_markdown_report

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "hit_rate_at_k",
    "compute_retrieval_metrics",
    "evaluate_faithfulness",
    "evaluate_relevance",
    "run_benchmark_eval",
    "generate_markdown_report"
]
