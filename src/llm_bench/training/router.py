"""Prompt router — predict which model to use based on the prompt.

Embeds prompts with sentence-transformers, trains a classifier to predict
the best model. No GPU required.

Usage:
    from llm_bench.training import PromptRouter
    router = PromptRouter()
    router.train(storage, task_name="summarize")
    best = router.predict("You are a summarizer.", "Summarize this article...")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _check_deps():
    try:
        import sklearn  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "Router training requires scikit-learn and sentence-transformers. "
            "Install with: pip install llm-bench[train]"
        )


class PromptRouter:
    """Lightweight prompt→model classifier.

    Embeds prompts with all-MiniLM-L6-v2 (384-dim, 80MB, CPU),
    trains an XGBoost or Random Forest classifier.

    Args:
        embedding_model: sentence-transformers model name. Default: all-MiniLM-L6-v2.
        classifier: "xgboost" or "random_forest". Falls back to RF if xgboost not installed.
        confidence_threshold: Below this, predict_with_confidence returns None.
    """

    def __init__(
        self,
        embedding_model: str = _EMBEDDING_MODEL,
        classifier: str = "random_forest",
        confidence_threshold: float = 0.6,
    ) -> None:
        self._embedding_model_name = embedding_model
        self._classifier_type = classifier
        self._confidence_threshold = confidence_threshold
        self._embedder = None
        self._classifier = None
        self._label_encoder = None
        self._is_trained = False

    def train(
        self,
        storage: Any,
        task_name: str | None = None,
        min_quality: float = 0.3,
        min_runs: int = 2,
    ) -> dict:
        """Train the router from stored benchmark results.

        Args:
            storage: Storage backend.
            task_name: Filter to a specific task.
            min_quality: Minimum quality for a result to be considered valid.
            min_runs: Minimum runs per (fixture, model) to be considered.

        Returns:
            Training metrics: {n_examples, n_classes, accuracy, classes}.
        """
        _check_deps()
        from sentence_transformers import SentenceTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder

        from llm_bench.training.export import export_for_router

        # Get training data
        data = export_for_router(storage, task_name=task_name, min_quality=min_quality, min_runs=min_runs)
        if len(data) < 5:
            raise ValueError(
                f"Need at least 5 training examples, got {len(data)}. "
                f"Run more benchmarks first."
            )

        # Embed prompts
        logger.info("Loading embedding model: %s", self._embedding_model_name)
        self._embedder = SentenceTransformer(self._embedding_model_name)

        # Use fixture_id as a proxy for prompt content since we may not have full prompts stored
        texts = [d.get("prompt", d["fixture_id"]) for d in data]
        labels = [d["best_model"] for d in data]

        logger.info("Embedding %d prompts...", len(texts))
        embeddings = self._embedder.encode(texts, show_progress_bar=False)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(labels)

        # Train classifier
        if self._classifier_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                self._classifier = XGBClassifier(
                    n_estimators=100, max_depth=6, use_label_encoder=False,
                    eval_metric="mlogloss", verbosity=0,
                )
            except ImportError:
                logger.warning("xgboost not installed, falling back to RandomForest")
                self._classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self._classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        self._classifier.fit(embeddings, y)
        self._is_trained = True

        # Cross-validation accuracy (if enough data)
        accuracy = 0.0
        if len(data) >= 10:
            n_splits = min(5, len(set(y)))
            if n_splits >= 2:
                scores = cross_val_score(self._classifier, embeddings, y, cv=n_splits)
                accuracy = float(scores.mean())

        classes = list(self._label_encoder.classes_)
        metrics = {
            "n_examples": len(data),
            "n_classes": len(classes),
            "accuracy": accuracy,
            "classes": classes,
        }
        logger.info("Router trained: %s", metrics)
        return metrics

    def predict(self, system_prompt: str, user_prompt: str) -> str | None:
        """Predict the best model for a prompt.

        Returns the model config name, or None if not trained.
        """
        if not self._is_trained:
            return None

        embedding = self._embedder.encode([f"{system_prompt}\n{user_prompt}"])
        pred_idx = self._classifier.predict(embedding)[0]
        return self._label_encoder.inverse_transform([pred_idx])[0]

    def predict_with_confidence(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str | None, float]:
        """Predict with confidence score.

        Returns (model_name, confidence). If confidence < threshold, returns (None, confidence).
        """
        if not self._is_trained:
            return None, 0.0

        embedding = self._embedder.encode([f"{system_prompt}\n{user_prompt}"])
        proba = self._classifier.predict_proba(embedding)[0]
        best_idx = proba.argmax()
        confidence = float(proba[best_idx])
        model_name = self._label_encoder.inverse_transform([best_idx])[0]

        if confidence < self._confidence_threshold:
            return None, confidence

        return model_name, confidence

    def save(self, path: str | Path) -> None:
        """Save trained router to disk."""
        if not self._is_trained:
            raise ValueError("Router not trained yet")

        import joblib

        path = Path(path)
        joblib.dump({
            "classifier": self._classifier,
            "label_encoder": self._label_encoder,
            "embedding_model": self._embedding_model_name,
            "classifier_type": self._classifier_type,
            "confidence_threshold": self._confidence_threshold,
        }, path)
        logger.info("Router saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> PromptRouter:
        """Load a trained router from disk."""
        _check_deps()
        import joblib
        from sentence_transformers import SentenceTransformer

        data = joblib.load(path)
        router = cls(
            embedding_model=data["embedding_model"],
            classifier=data["classifier_type"],
            confidence_threshold=data["confidence_threshold"],
        )
        router._classifier = data["classifier"]
        router._label_encoder = data["label_encoder"]
        router._embedder = SentenceTransformer(data["embedding_model"])
        router._is_trained = True
        return router

    @property
    def is_trained(self) -> bool:
        return self._is_trained
