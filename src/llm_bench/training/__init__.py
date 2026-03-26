"""Training module — train routers, predictors, and prepare fine-tuning data from benchmark results.

Training classes require optional dependencies:
    pip install llm-bench[train]       # router + Ridge predictor
    pip install llm-bench[train-deep]  # DistilBERT predictor

Export functions work without extra deps.
"""

from llm_bench.training.export import export_for_predictor, export_for_router, to_openai_jsonl


def __getattr__(name: str):
    """Lazy imports for classes that require optional dependencies."""
    if name == "PromptRouter":
        from llm_bench.training.router import PromptRouter
        return PromptRouter
    if name == "QualityPredictor":
        from llm_bench.training.predictor import QualityPredictor
        return QualityPredictor
    raise AttributeError(f"module 'llm_bench.training' has no attribute {name!r}")


__all__ = [
    "PromptRouter",
    "QualityPredictor",
    "export_for_predictor",
    "export_for_router",
    "to_openai_jsonl",
]
