"""LLM access: schema preparation, coercion, JSON extraction and the Ollama client (docs/04 §1)."""

from caracal.llm.client import CallMeta, LLMClient, ToolResult
from caracal.llm.jsonx import extract_json
from caracal.llm.schema import DEFAULT_SYNONYMS, coerce, prepare_schema

__all__ = [
    "DEFAULT_SYNONYMS",
    "CallMeta",
    "LLMClient",
    "ToolResult",
    "coerce",
    "extract_json",
    "prepare_schema",
]
