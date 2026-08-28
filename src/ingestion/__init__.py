"""Ingestion module for PDF extraction, structured chunking, and index construction."""
from .pdf_parser import extract_paper_pages
from .chunker import chunk_paper
from .indexer import build_indices, load_or_build_indices

__all__ = ["extract_paper_pages", "chunk_paper", "build_indices", "load_or_build_indices"]
