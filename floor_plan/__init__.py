"""Floor-plan analysis package."""

from .analyzer import (
    DEFAULT_MODEL,
    SCORE_CRITERIA,
    FloorPlanAnalysis,
    analyze_floor_plan,
    build_analysis_prompt,
    build_extraction_prompt,
    load_knowledge,
    parse_structure_response,
)
from .providers import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    FallbackClient,
    GeminiClient,
    OllamaClient,
    create_analysis_client,
)

__all__ = [
    "DEFAULT_MODEL",
    "SCORE_CRITERIA",
    "FloorPlanAnalysis",
    "analyze_floor_plan",
    "build_analysis_prompt",
    "build_extraction_prompt",
    "load_knowledge",
    "parse_structure_response",
]
