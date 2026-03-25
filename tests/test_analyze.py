"""Tests for the analysis engine."""

from llm_bench.analyze import (
    aggregate,
    cost_per_success,
    format_table,
    get_recommendations,
    pareto_frontier,
    recommend,
)


def _make_result(task="test", config="model-a", quality=0.8, cost=0.01, error=None):
    return {
        "task_name": task,
        "config_name": config,
        "quality_score": quality,
        "cost_usd": cost,
        "latency_s": 1.5,
        "error": error,
    }


class TestAggregate:
    def test_basic_aggregation(self):
        results = [
            _make_result(config="a", quality=0.8, cost=0.01),
            _make_result(config="a", quality=0.6, cost=0.02),
            _make_result(config="b", quality=0.9, cost=0.05),
        ]
        agg = aggregate(results)
        assert "test" in agg
        assert "a" in agg["test"]
        assert "b" in agg["test"]
        assert abs(agg["test"]["a"]["mean_quality"] - 0.7) < 0.01
        assert abs(agg["test"]["b"]["mean_quality"] - 0.9) < 0.01
        assert agg["test"]["a"]["n_runs"] == 2
        assert agg["test"]["b"]["n_runs"] == 1

    def test_error_handling(self):
        results = [
            _make_result(config="a", quality=0.8),
            _make_result(config="a", quality=0.0, error="some error"),
        ]
        agg = aggregate(results)
        assert agg["test"]["a"]["n_errors"] == 1
        assert agg["test"]["a"]["n_runs"] == 2
        # Mean quality should only count non-error runs
        assert abs(agg["test"]["a"]["mean_quality"] - 0.8) < 0.01

    def test_transient_errors_excluded(self):
        results = [
            _make_result(config="a", quality=0.8),
            _make_result(config="a", quality=0.0, error="429 rate limit exceeded"),
        ]
        agg = aggregate(results)
        assert agg["test"]["a"]["n_runs"] == 1  # transient excluded from n_runs
        assert agg["test"]["a"]["n_rate_limited"] == 1

    def test_multiple_tasks(self):
        results = [
            _make_result(task="gen", config="a", quality=0.8),
            _make_result(task="eval", config="a", quality=0.9),
        ]
        agg = aggregate(results)
        assert "gen" in agg
        assert "eval" in agg

    def test_sub_scores_aggregation(self):
        results = [
            {"task_name": "t", "config_name": "a", "quality_score": 0.8,
             "cost_usd": 0.01, "latency_s": 1.0, "error": None,
             "sub_scores": {"accuracy": 0.9, "fluency": 0.7}},
            {"task_name": "t", "config_name": "a", "quality_score": 0.6,
             "cost_usd": 0.01, "latency_s": 1.0, "error": None,
             "sub_scores": {"accuracy": 0.7, "fluency": 0.5}},
        ]
        agg = aggregate(results)
        assert abs(agg["t"]["a"]["sub_scores"]["accuracy"] - 0.8) < 0.01
        assert abs(agg["t"]["a"]["sub_scores"]["fluency"] - 0.6) < 0.01

    def test_empty_results(self):
        assert aggregate([]) == {}


class TestParetoFrontier:
    def test_simple_frontier(self):
        configs = {
            "cheap_bad": {"mean_quality": 0.3, "mean_cost": 0.001},
            "expensive_good": {"mean_quality": 0.9, "mean_cost": 0.05},
            "dominated": {"mean_quality": 0.3, "mean_cost": 0.05},  # worse quality AND cost
        }
        frontier = pareto_frontier(configs)
        assert "cheap_bad" in frontier
        assert "expensive_good" in frontier
        assert "dominated" not in frontier

    def test_single_config(self):
        configs = {"only": {"mean_quality": 0.5, "mean_cost": 0.01}}
        assert pareto_frontier(configs) == {"only"}

    def test_all_on_frontier(self):
        configs = {
            "a": {"mean_quality": 0.9, "mean_cost": 0.05},
            "b": {"mean_quality": 0.5, "mean_cost": 0.01},
            "c": {"mean_quality": 0.3, "mean_cost": 0.001},
        }
        frontier = pareto_frontier(configs)
        assert frontier == {"a", "b", "c"}


class TestCostPerSuccess:
    def test_perfect_quality(self):
        stats = {"mean_quality": 1.0, "mean_cost": 0.01}
        assert abs(cost_per_success(stats) - 0.01) < 0.001

    def test_zero_quality(self):
        stats = {"mean_quality": 0.0, "mean_cost": 0.01}
        assert cost_per_success(stats) == float("inf")

    def test_includes_time_waste(self):
        stats = {"mean_quality": 0.5, "mean_cost": 0.01}
        cps = cost_per_success(stats, cycle_time_cost=0.002)
        # With 50% quality: cost_per_success = (0.01 + 0.002 * 1) / 0.5 = 0.024
        assert abs(cps - 0.024) < 0.001


class TestRecommend:
    def test_selects_cheapest_above_floor(self):
        configs = {
            "cheap": {"mean_quality": 0.6, "mean_cost": 0.001},
            "expensive": {"mean_quality": 0.9, "mean_cost": 0.05},
        }
        name, stats = recommend("test", configs, quality_floor=0.5)
        assert name == "cheap"

    def test_excludes_below_floor(self):
        configs = {
            "bad": {"mean_quality": 0.3, "mean_cost": 0.001},
            "good": {"mean_quality": 0.8, "mean_cost": 0.05},
        }
        name, stats = recommend("test", configs, quality_floor=0.5)
        assert name == "good"

    def test_no_candidates(self):
        configs = {
            "bad": {"mean_quality": 0.1, "mean_cost": 0.001},
        }
        name, stats = recommend("test", configs, quality_floor=0.5)
        assert name is None
        assert stats is None


class TestGetRecommendations:
    def test_multi_task(self):
        results = [
            _make_result(task="gen", config="a", quality=0.8, cost=0.01),
            _make_result(task="gen", config="b", quality=0.6, cost=0.001),
            _make_result(task="eval", config="a", quality=0.9, cost=0.01),
        ]
        recs = get_recommendations(results, default_floor=0.5)
        assert "gen" in recs
        assert "eval" in recs

    def test_custom_floors(self):
        results = [
            _make_result(task="gen", config="a", quality=0.6, cost=0.01),
        ]
        recs = get_recommendations(results, quality_floors={"gen": 0.7})
        assert "gen" not in recs  # 0.6 < floor of 0.7

    def test_empty(self):
        assert get_recommendations([]) == {}


class TestFormatTable:
    def test_produces_output(self):
        configs = {
            "a": {"mean_quality": 0.8, "std_quality": 0.1, "mean_cost": 0.01,
                   "mean_latency": 1.5, "n_runs": 10, "n_errors": 0, "n_rate_limited": 0,
                   "error_rate": 0.0},
        }
        output = format_table("test", configs)
        assert "TEST" in output
        assert "a" in output
        assert "quality" in output
