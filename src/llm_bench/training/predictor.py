"""Quality predictor — predict quality score from response text.

Two tiers:
  - "ridge": scikit-learn Ridge regression on embeddings (seconds to train, CPU)
  - "distilbert": fine-tuned DistilBERT regression (~30 min on CPU)

Usage:
    from llm_bench.training import QualityPredictor
    predictor = QualityPredictor(method="ridge")
    predictor.train(storage, task_name="summarize")
    score = predictor.predict("The article discusses...")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _check_train_deps():
    try:
        import sklearn  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "Quality predictor requires scikit-learn and sentence-transformers. "
            "Install with: pip install llm-bench[train]"
        )


def _check_deep_deps():
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "DistilBERT predictor requires transformers and torch. "
            "Install with: pip install llm-bench[train-deep]"
        )


class QualityPredictor:
    """Predicts quality score from response text.

    Args:
        method: "ridge" (fast, CPU) or "distilbert" (more accurate, slower).
        embedding_model: sentence-transformers model for Ridge method.
    """

    def __init__(
        self,
        method: str = "ridge",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        if method not in ("ridge", "distilbert"):
            raise ValueError(f"Unknown method: {method!r}. Use 'ridge' or 'distilbert'.")
        self._method = method
        self._embedding_model_name = embedding_model
        self._model = None
        self._embedder = None
        self._is_trained = False
        self._metrics: dict = {}

    def train(
        self,
        storage: Any,
        task_name: str | None = None,
        epochs: int = 3,
        batch_size: int = 16,
    ) -> dict:
        """Train the quality predictor from stored benchmark results.

        Args:
            storage: Storage backend.
            task_name: Filter to a specific task.
            epochs: Training epochs (only for distilbert method).
            batch_size: Batch size (only for distilbert method).

        Returns:
            Training metrics dict.
        """
        from llm_bench.training.export import export_for_predictor

        data = export_for_predictor(storage, task_name=task_name)
        if len(data) < 10:
            raise ValueError(
                f"Need at least 10 training examples with raw_response, got {len(data)}. "
                f"Run more benchmarks first."
            )

        if self._method == "ridge":
            return self._train_ridge(data)
        else:
            return self._train_distilbert(data, epochs, batch_size)

    def _train_ridge(self, data: list[dict]) -> dict:
        _check_train_deps()
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        texts = [d["text"] for d in data]
        labels = np.array([d["label"] for d in data])

        logger.info("Loading embedding model for Ridge predictor...")
        self._embedder = SentenceTransformer(self._embedding_model_name)

        logger.info("Embedding %d responses...", len(texts))
        embeddings = self._embedder.encode(texts, show_progress_bar=False)

        self._model = Ridge(alpha=1.0)
        self._model.fit(embeddings, labels)
        self._is_trained = True

        # Cross-validation R² score
        r2 = 0.0
        if len(data) >= 20:
            scores = cross_val_score(self._model, embeddings, labels, cv=5, scoring="r2")
            r2 = float(scores.mean())

        # Mean absolute error
        preds = self._model.predict(embeddings)
        mae = float(np.mean(np.abs(preds - labels)))

        self._metrics = {
            "method": "ridge",
            "n_examples": len(data),
            "r2_cv": r2,
            "mae_train": mae,
        }
        logger.info("Ridge predictor trained: %s", self._metrics)
        return self._metrics

    def _train_distilbert(self, data: list[dict], epochs: int, batch_size: int) -> dict:
        _check_deep_deps()
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        texts = [d["text"][:512] for d in data]  # truncate for BERT
        labels = [d["label"] for d in data]

        model_name = "distilbert-base-uncased"
        logger.info("Loading %s for quality predictor...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, problem_type="regression"
        )

        # Create dataset
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

        class QualityDataset(torch.utils.data.Dataset):
            def __init__(self, enc, labs):
                self.enc = enc
                self.labs = labs

            def __len__(self):
                return len(self.labs)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
                item["labels"] = torch.tensor(self.labs[idx], dtype=torch.float32)
                return item

        dataset = QualityDataset(encodings, labels)

        # Split train/val
        n_val = max(1, len(dataset) // 5)
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

        training_args = TrainingArguments(
            output_dir="/tmp/llm-bench-predictor",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )

        logger.info("Training DistilBERT predictor (%d examples, %d epochs)...", len(data), epochs)
        trainer.train()

        self._model = model
        self._embedder = tokenizer  # reuse tokenizer reference
        self._is_trained = True

        # Evaluate
        eval_result = trainer.evaluate()
        self._metrics = {
            "method": "distilbert",
            "n_examples": len(data),
            "eval_loss": eval_result.get("eval_loss", 0),
        }
        logger.info("DistilBERT predictor trained: %s", self._metrics)
        return self._metrics

    def predict(self, response_text: str) -> float:
        """Predict quality score for a response.

        Args:
            response_text: The LLM response to evaluate.

        Returns:
            Predicted quality score (0-1, clamped).
        """
        if not self._is_trained:
            raise ValueError("Predictor not trained yet")

        if self._method == "ridge":
            embedding = self._embedder.encode([response_text])
            pred = float(self._model.predict(embedding)[0])
        else:
            import torch
            inputs = self._embedder(
                response_text[:512], return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                output = self._model(**inputs)
            pred = float(output.logits.squeeze().item())

        return max(0.0, min(1.0, pred))

    def save(self, path: str | Path) -> None:
        """Save trained predictor to disk."""
        if not self._is_trained:
            raise ValueError("Predictor not trained yet")

        path = Path(path)

        if self._method == "ridge":
            import joblib
            joblib.dump({
                "method": "ridge",
                "model": self._model,
                "embedding_model": self._embedding_model_name,
                "metrics": self._metrics,
            }, path)
        else:
            # Save DistilBERT model + tokenizer to directory
            path.mkdir(parents=True, exist_ok=True)
            self._model.save_pretrained(path)
            self._embedder.save_pretrained(path)
            import json
            (path / "predictor_meta.json").write_text(json.dumps(self._metrics))

        logger.info("Predictor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> QualityPredictor:
        """Load a trained predictor from disk."""
        path = Path(path)

        if path.is_file():
            # Ridge model (single file)
            _check_train_deps()
            import joblib
            from sentence_transformers import SentenceTransformer

            data = joblib.load(path)
            predictor = cls(method="ridge", embedding_model=data["embedding_model"])
            predictor._model = data["model"]
            predictor._embedder = SentenceTransformer(data["embedding_model"])
            predictor._metrics = data.get("metrics", {})
            predictor._is_trained = True
            return predictor
        else:
            # DistilBERT model (directory)
            _check_deep_deps()
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import json

            predictor = cls(method="distilbert")
            predictor._model = AutoModelForSequenceClassification.from_pretrained(path)
            predictor._embedder = AutoTokenizer.from_pretrained(path)
            meta_path = path / "predictor_meta.json"
            if meta_path.exists():
                predictor._metrics = json.loads(meta_path.read_text())
            predictor._is_trained = True
            return predictor

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def metrics(self) -> dict:
        return self._metrics
