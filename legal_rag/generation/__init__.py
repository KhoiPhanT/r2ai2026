from legal_rag.generation.ollama import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, OllamaConfig, OllamaError
from legal_rag.generation.evidence import (
    EvidenceAnswer,
    EvidenceBlock,
    articles_from_used_evidence,
    build_evidence_blocks,
    generate_evidence_answer,
)
from legal_rag.generation.ollama import generate_ollama_answer
from legal_rag.generation.template import generate_grounded_answer

__all__ = [
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_URL",
    "EvidenceAnswer",
    "EvidenceBlock",
    "OllamaConfig",
    "OllamaError",
    "articles_from_used_evidence",
    "build_evidence_blocks",
    "generate_evidence_answer",
    "generate_grounded_answer",
    "generate_ollama_answer",
]
