"""Tests for storage backends."""

import tempfile
from pathlib import Path

from llm_bench.storage.json_file import JSONFileStorage
from llm_bench.storage.sqlite import SQLiteStorage
from llm_bench.types import BenchmarkResult


def _make_result(**kwargs):
    defaults = {
        "task_name": "test",
        "config_name": "haiku",
        "fixture_id": "fixture_1",
        "quality_score": 0.8,
        "sub_scores": {"accuracy": 0.9},
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.001,
        "latency_s": 1.5,
        "error": None,
    }
    defaults.update(kwargs)
    return BenchmarkResult(**defaults)


class TestSQLiteStorage:
    def test_save_and_load(self, tmp_path):
        db = tmp_path / "test.db"
        storage = SQLiteStorage(db)

        result = _make_result()
        storage.save_result(result)

        loaded = storage.load_results()
        assert len(loaded) == 1
        assert loaded[0]["task_name"] == "test"
        assert loaded[0]["config_name"] == "haiku"
        assert abs(loaded[0]["quality_score"] - 0.8) < 0.01

    def test_save_multiple(self, tmp_path):
        storage = SQLiteStorage(tmp_path / "test.db")

        results = [
            _make_result(config_name="a"),
            _make_result(config_name="b"),
            _make_result(config_name="c"),
        ]
        storage.save_results(results)

        loaded = storage.load_results()
        assert len(loaded) == 3

    def test_filter_by_task(self, tmp_path):
        storage = SQLiteStorage(tmp_path / "test.db")

        storage.save_result(_make_result(task_name="gen"))
        storage.save_result(_make_result(task_name="eval"))

        gen_results = storage.load_results(task_name="gen")
        assert len(gen_results) == 1
        assert gen_results[0]["task_name"] == "gen"

    def test_sub_scores_json(self, tmp_path):
        storage = SQLiteStorage(tmp_path / "test.db")

        storage.save_result(_make_result(sub_scores={"accuracy": 0.9, "fluency": 0.7}))

        loaded = storage.load_results()
        assert loaded[0]["sub_scores"]["accuracy"] == 0.9
        assert loaded[0]["sub_scores"]["fluency"] == 0.7

    def test_overrides(self, tmp_path):
        storage = SQLiteStorage(tmp_path / "test.db")

        assert storage.get_override("gen") is None

        storage.set_override("gen", "haiku")
        assert storage.get_override("gen") == "haiku"

        storage.set_override("gen", None)
        assert storage.get_override("gen") is None

    def test_get_all_overrides(self, tmp_path):
        storage = SQLiteStorage(tmp_path / "test.db")

        storage.set_override("gen", "haiku")
        storage.set_override("eval", "sonnet")

        overrides = storage.get_all_overrides()
        assert overrides == {"gen": "haiku", "eval": "sonnet"}


class TestJSONFileStorage:
    def test_save_and_load(self, tmp_path):
        storage = JSONFileStorage(tmp_path / "results")

        result = _make_result()
        storage.save_result(result)

        loaded = storage.load_results()
        assert len(loaded) == 1
        assert loaded[0]["task_name"] == "test"

    def test_filter_by_task(self, tmp_path):
        storage = JSONFileStorage(tmp_path / "results")

        storage.save_result(_make_result(task_name="gen"))
        storage.save_result(_make_result(task_name="eval"))

        gen_results = storage.load_results(task_name="gen")
        assert len(gen_results) == 1

    def test_overrides(self, tmp_path):
        storage = JSONFileStorage(tmp_path / "results")

        assert storage.get_override("gen") is None

        storage.set_override("gen", "haiku")
        assert storage.get_override("gen") == "haiku"

        storage.set_override("gen", None)
        assert storage.get_override("gen") is None
