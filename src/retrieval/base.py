"""Base retriever definitions and data structures."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class RetrievedChunk:
    """Represents a retrieved text chunk with provenance metadata and score."""
    chunk_id: str
    paper_id: str
    paper_title: str
    page: int
    section: str
    text: str
    score: float
    retrieval_strategy: str
    rank: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def formatted_citation(self) -> str:
        return f"[{self.chunk_id}] ({self.paper_title}, Page {self.page}, Section '{self.section}')"

class BaseRetriever(ABC):
    """Abstract interface for all retrieval strategies."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        """Retrieve the top-k chunks for a given query."""
        pass
