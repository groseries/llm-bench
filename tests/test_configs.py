"""Tests for model configs."""

from llm_bench.configs import ALL_CONFIGS, CONFIG_BY_NAME, ModelConfig, ModelRegistry


class TestModelConfig:
    def test_frozen(self):
        config = ModelConfig("test", "provider/model")
        try:
            config.name = "changed"
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_defaults(self):
        config = ModelConfig("test", "provider/model")
        assert config.thinking_budget == 0


class TestBuiltInConfigs:
    def test_all_configs_not_empty(self):
        assert len(ALL_CONFIGS) > 0

    def test_unique_names(self):
        names = [c.name for c in ALL_CONFIGS]
        assert len(names) == len(set(names))

    def test_config_by_name(self):
        assert "haiku" in CONFIG_BY_NAME
        assert CONFIG_BY_NAME["haiku"].model == "anthropic/claude-haiku-4.5"

    def test_all_have_openrouter_slugs(self):
        for config in ALL_CONFIGS:
            assert "/" in config.model, f"{config.name} missing provider prefix in model slug"


class TestModelRegistry:
    def test_defaults_loaded(self):
        registry = ModelRegistry()
        assert len(registry.all_configs) == len(ALL_CONFIGS)

    def test_no_defaults(self):
        registry = ModelRegistry(include_defaults=False)
        assert len(registry.all_configs) == 0

    def test_add_custom(self):
        registry = ModelRegistry(include_defaults=False)
        registry.add(ModelConfig("custom", "my/model"))
        assert registry.get("custom") is not None
        assert registry.get("custom").model == "my/model"

    def test_remove(self):
        registry = ModelRegistry()
        initial_count = len(registry.all_configs)
        registry.remove("haiku")
        assert len(registry.all_configs) == initial_count - 1
        assert registry.get("haiku") is None

    def test_subset(self):
        registry = ModelRegistry()
        subset = registry.subset(["haiku", "sonnet"])
        assert len(subset) == 2
        assert subset[0].name == "haiku"
        assert subset[1].name == "sonnet"

    def test_subset_missing(self):
        registry = ModelRegistry()
        subset = registry.subset(["haiku", "nonexistent"])
        assert len(subset) == 1
