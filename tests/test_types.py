"""Tests for core types."""

from llm_bench.types import BenchmarkResult, Fixture, QualityScorer, SimpleFixture, TaskDefinition


class TestSimpleFixture:
    def test_conforms_to_protocol(self):
        fixture = SimpleFixture(
            fixture_id="test",
            system_prompt="system",
            user_prompt="user",
        )
        assert isinstance(fixture, Fixture)

    def test_attributes(self):
        fixture = SimpleFixture("id", "sys", "usr")
        assert fixture.fixture_id == "id"
        assert fixture.system_prompt == "sys"
        assert fixture.user_prompt == "usr"


class TestBenchmarkResult:
    def test_to_dict(self):
        result = BenchmarkResult(
            task_name="test",
            config_name="haiku",
            fixture_id="f1",
            quality_score=0.8,
        )
        d = result.to_dict()
        assert d["task_name"] == "test"
        assert d["quality_score"] == 0.8
        assert "timestamp" in d

    def test_defaults(self):
        result = BenchmarkResult(
            task_name="t", config_name="c", fixture_id="f", quality_score=0.5
        )
        assert result.input_tokens == 0
        assert result.cost_usd == 0.0
        assert result.error is None
        assert result.sub_scores == {}


class TestQualityScorerProtocol:
    def test_class_conforms(self):
        class MyScorer:
            def score(self, response, fixture, **kwargs):
                return {"score": 0.5}

        assert isinstance(MyScorer(), QualityScorer)

    def test_scorer_call(self):
        class MyScorer:
            def score(self, response, fixture, **kwargs):
                return {"score": 0.8, "accuracy": 0.9}

        fixture = SimpleFixture("f1", "sys", "usr")
        result = MyScorer().score("test response", fixture)
        assert result["score"] == 0.8
        assert result["accuracy"] == 0.9


class TestTaskDefinition:
    def test_creation(self):
        class S:
            def score(self, response, fixture, **kwargs):
                return {"score": 0.5}

        task = TaskDefinition(
            name="test",
            scorer=S(),
            fixtures=[SimpleFixture("f1", "s", "u")],
            quality_floor=0.7,
        )
        assert task.name == "test"
        assert task.quality_floor == 0.7
        assert len(task.fixtures) == 1

    def test_defaults(self):
        class S:
            def score(self, response, fixture, **kwargs):
                return {"score": 0.5}

        task = TaskDefinition(name="t", scorer=S(), fixtures=[])
        assert task.quality_floor == 0.5
        assert task.models is None
