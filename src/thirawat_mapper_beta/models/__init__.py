"""Model wrappers used in the thirawat_mapper_beta toolkit."""

from .embedder import SapBERTEmbedder
from .rag_llm import (
    BaseLLMClient,
    CloudflareConfig,
    CloudflareLLMClient,
    LlamaCppConfig,
    LlamaCppLLMClient,
    LlamaCppServerConfig,
    LlamaCppServerLLMClient,
    OllamaConfig,
    OllamaLLMClient,
    OpenRouterConfig,
    OpenRouterLLMClient,
)
from .rag_pipeline import RAGPipeline, RagRanking
from .rag_prompt import RAGPromptBuilder, RagCandidate, TemplatePromptBuilder, to_candidates
from .reranker import DEFAULT_RERANKER_ID, ThirawatReranker

__all__ = [
    "BaseLLMClient",
    "CloudflareConfig",
    "CloudflareLLMClient",
    "LlamaCppConfig",
    "LlamaCppLLMClient",
    "LlamaCppServerConfig",
    "LlamaCppServerLLMClient",
    "OllamaConfig",
    "OllamaLLMClient",
    "OpenRouterConfig",
    "OpenRouterLLMClient",
    "RAGPipeline",
    "RagCandidate",
    "RAGPromptBuilder",
    "RagRanking",
    "SapBERTEmbedder",
    "TemplatePromptBuilder",
    "ThirawatReranker",
    "DEFAULT_RERANKER_ID",
    "to_candidates",
]
