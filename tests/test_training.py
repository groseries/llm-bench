"""Tests for the training module (export, router, predictor)."""

import json
from pathlib import Path

import pytest

from llm_bench.storage.sqlite import SQLiteStorage
from llm_bench.types import BenchmarkResult


def _populate_storage(storage, n_fixtures=10, n_models=3):
    """Create synthetic benchmark data for training tests."""
    models = ["haiku", "gemini-2.5-flash", "devstral-small"][:n_models]
    # Each model has different quality characteristics
    quality_by_model = {
        "haiku": 0.85,
        "gemini-2.5-flash": 0.70,
        "devstral-small": 0.55,
    }
    cost_by_model = {
        "haiku": 0.005,
        "gemini-2.5-flash": 0.001,
        "devstral-small": 0.0003,
    }

    for i in range(n_fixtures):
        for model in models:
            for run in range(3):  # 3 runs per combo
                base_quality = quality_by_model.get(model, 0.5)
                # Add some variance
                quality = base_quality + (run - 1) * 0.05
                storage.save_result(BenchmarkResult(
                    task_name="test_task",
                    config_name=model,
                    fixture_id=f"fixture_{i}",
                    quality_score=max(0.0, min(1.0, quality)),
                    sub_scores={"accuracy": quality, "fluency": quality * 0.9},
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd=cost_by_model.get(model, 0.01),
                    latency_s=1.5,
                    raw_response=f"This is a test response from {model} for fixture {i}.",
                ))


class TestExport:
    def test_for_router(self, tmp_path):
        from llm_bench.training.export import export_for_router

        storage = SQLiteStorage(tmp_path / "test.db")
        _populate_storage(storage)

        data = export_for_router(storage, task_name="test_task")
        assert len(data) > 0
        for item in data:
            assert "task_name" in item
            assert "fixture_id" in item
            assert "best_model" in item
            assert "quality" in item
            assert item["quality"] > 0

    def test_for_router_min_quality(self, tmp_path):
        from llm_bench.training.export import export_for_router

        storage = SQLiteStorage(tmp_path / "test.db")
        _populate_storage(storage)

        # With high floor, fewer results
        data_high = export_for_router(storage, min_quality=0.8)
        data_low = export_for_router(storage, min_quality=0.3)
        assert len(data_low) >= len(data_high)

    def test_for_predictor(self, tmp_path):
        from llm_bench.training.export import export_for_predictor

        storage = SQLiteStorage(tmp_path / "test.db")
        _populate_storage(storage)

        data = export_for_predictor(storage)
        assert len(data) > 0
        for item in data:
            assert "text" in item
            assert "label" in item
            assert 0.0 <= item["label"] <= 1.0

    def test_to_openai_jsonl(self, tmp_path):
        from llm_bench.training.export import to_openai_jsonl

        storage = SQLiteStorage(tmp_path / "test.db")
        _populate_storage(storage)

        output_path = tmp_path / "finetune.jsonl"
        count = to_openai_jsonl(storage, output_path, task_name="test_task", min_quality=0.5)
        assert count > 0
        assert output_path.exists()

        # Verify JSONL format
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                assert "messages" in data

    def test_empty_storage(self, tmp_path):
        from llm_bench.training.export import export_for_router

        storage = SQLiteStorage(tmp_path / "test.db")
        data = export_for_router(storage)
        assert data == []


class TestDistill:
    def test_prepare_openai(self, tmp_path):
        from llm_bench.training.distill import prepare_openai

        storage = SQLiteStorage(tmp_path / "test.db")
        _populate_storage(storage)

        output = tmp_path / "ft.jsonl"
        count = prepare_openai(storage, "test_task", output, min_quality=0.5)
        assert count > 0
        assert output.exists()

    def test_prepare_vertex(self, tmp_path):
        from llm_bench.training.distill import prepare_vertex

        storage = SQLiteStorage(tmp_path / "test.db")
        _populate_storage(storage)

        output = tmp_path / "ft_vertex.jsonl"
        count = prepare_vertex(storage, "test_task", output, min_quality=0.5)
        assert count > 0

        with open(output) as f:
            for line in f:
                data = json.loads(line)
                assert "contents" in data

    def test_source_model_filter(self, tmp_path):
        from llm_bench.training.distill import prepare_openai

        storage = SQLiteStorage(tmp_path / "test.db")
        _populate_storage(storage)

        output = tmp_path / "ft.jsonl"
        count = prepare_openai(
            storage, "test_task", output,
            min_quality=0.5, source_model="haiku",
        )
        assert count > 0


# Router and predictor tests require optional deps (scikit-learn, sentence-transformers)
# Skip if not installed

@pytest.fixture
def has_train_deps():
    try:
        import sklearn  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        pytest.skip("scikit-learn and sentence-transformers not installed")


class TestPromptRouter:
    def test_train_and_predict(self, tmp_path, has_train_deps):
        from llm_bench.training.router import PromptRouter

        storage = SQLiteStorage(tmp_path / "test.db")
        _populate_storage(storage, n_fixtures=10, n_models=3)

        router = PromptRouter()
        metrics = router.train(storage, task_name="test_task", min_runs=2)

        assert metrics["n_examples"] > 0
        assert metrics["n_classes"] > 0
        assert router.is_trained

        # Predict
        result = router.predict("You are a helpful assistant.", "Test prompt")
        assert result in ["haiku", "gemini-2.5-flash", "devstral-small"]

    def test_predict_with_confidence(self, tmp_path, has_train_deps):
        from llm_bench.training.router import PromptRouter

        storage = SQLiteStorage(tmp_path / "test.db")
        _populate_storage(storage, n_fixtures=10, n_models=3)

        router = PromptRouter(confidence_threshold=0.0)  # low threshold so we always get result
        router.train(storage, task_name="test_task", min_runs=2)

        model, confidence = router.predict_with_confidence("System", "User prompt")
        assert model is not None
        assert 0.0 <= confidence <= 1.0

    def test_save_and_load(self, tmp_path, has_train_deps):
        from llm_bench.training.router import PromptRouter

        storage = SQLiteStorage(tmp_path / "test.db")
        _populate_storage(storage, n_fixtures=10, n_models=3)

        router = PromptRouter()
        router.train(storage, task_name="test_task", min_runs=2)

        save_path = tmp_path / "router.pkl"
        router.save(save_path)
        assert save_path.exists()

        loaded = PromptRouter.load(save_path)
        assert loaded.is_trained
        result = loaded.predict("System", "Test")
        assert result in ["haiku", "gemini-2.5-flash", "devstral-small"]

    def test_too_few_examples(self, tmp_path, has_train_deps):
        from llm_bench.training.router import PromptRouter

        storage = SQLiteStorage(tmp_path / "test.db")
        # Only 1 fixture = too few for training
        _populate_storage(storage, n_fixtures=1, n_models=1)

        router = PromptRouter()
        with pytest.raises(ValueError, match="at least 5"):
            router.train(storage, min_runs=2)


class TestQualityPredictor:
    def test_ridge_train_and_predict(self, tmp_path, has_train_deps):
        from llm_bench.training.predictor import QualityPredictor

        storage = SQLiteStorage(tmp_path / "test.db")
        _populate_storage(storage, n_fixtures=10, n_models=3)

        predictor = QualityPredictor(method="ridge")
        metrics = predictor.train(storage, task_name="test_task")

        assert metrics["method"] == "ridge"
        assert metrics["n_examples"] > 0
        assert predictor.is_trained

        score = predictor.predict("This is a test response.")
        assert 0.0 <= score <= 1.0

    def test_ridge_save_and_load(self, tmp_path, has_train_deps):
        from llm_bench.training.predictor import QualityPredictor

        storage = SQLiteStorage(tmp_path / "test.db")
        _populate_storage(storage, n_fixtures=10, n_models=3)

        predictor = QualityPredictor(method="ridge")
        predictor.train(storage, task_name="test_task")

        save_path = tmp_path / "predictor.pkl"
        predictor.save(save_path)

        loaded = QualityPredictor.load(save_path)
        assert loaded.is_trained
        score = loaded.predict("Test response")
        assert 0.0 <= score <= 1.0

    def test_not_trained_raises(self, tmp_path):
        from llm_bench.training.predictor import QualityPredictor

        predictor = QualityPredictor(method="ridge")
        with pytest.raises(ValueError, match="not trained"):
            predictor.predict("test")
