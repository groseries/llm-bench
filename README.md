# llm-bench

**Benchmark LLM models on quality-per-cost and auto-select the optimal model per task.**

No existing tool connects custom quality evaluation to production model routing. Eval platforms (Langfuse, Braintrust) score quality but don't route calls. Routers (LiteLLM, OpenRouter auto-router) dispatch calls but don't benchmark with your quality metrics. llm-bench is the bridge: continuously benchmark models with your own scorers, then auto-select the cheapest model that meets your quality bar.

## How It Works

1. **Define tasks** — each LLM use case (summarize, classify, generate_code) gets a quality scorer and test fixtures
2. **Benchmark** — run your fixtures through N models via OpenRouter, score each response with your scorer
3. **Analyze** — Pareto frontier identifies models that are not dominated on both quality AND cost
4. **Auto-select** — the cheapest model above your quality floor wins, factoring in time waste from failures
5. **Re-evaluate** — continuous background benchmarking keeps recommendations fresh as models update

## Quickstart

```bash
pip install llm-bench
```

### Interactive setup

```bash
llm-bench init
```

The wizard walks you through defining tasks, selecting models, and generates scorer/fixture templates.

### Programmatic usage

```python
import llm_bench

# 1. Define a scorer
class MySummarizationScorer:
    def score(self, response: str, fixture, **kwargs) -> dict:
        has_key_points = all(kp in response.lower() for kp in ["climate", "2030"])
        is_concise = len(response.split()) < 200
        score = (0.7 if has_key_points else 0.0) + (0.3 if is_concise else 0.0)
        return {
            "score": score,
            "has_key_points": float(has_key_points),
            "is_concise": float(is_concise),
        }

# 2. Define fixtures
fixtures = [
    llm_bench.SimpleFixture(
        fixture_id="article_1",
        system_prompt="Summarize the following article concisely.",
        user_prompt="Article text about climate change targets for 2030...",
    ),
]

# 3. Create a task
task = llm_bench.TaskDefinition(
    name="summarize",
    scorer=MySummarizationScorer(),
    fixtures=fixtures,
    quality_floor=0.6,
)

# 4. Set up client + storage + runner
client = llm_bench.Client(api_key="sk-or-v1-your-key")
storage = llm_bench.SQLiteStorage()
runner = llm_bench.BenchmarkRunner(client=client, tasks=[task], storage=storage)

# 5. Run benchmarks
results = runner.run_all(n_runs=3)

# 6. Get the best model
selector = llm_bench.AutoSelector(storage=storage, tasks=[task])
recs = selector.refresh()
print(recs)  # {"summarize": "gemini-2.5-flash"}

# 7. Use in production
best = selector.get_best_model("summarize")
response = client.call(best, system_prompt, user_message)
```

## Concepts

### Tasks, Scorers, and Fixtures

A **Task** is a distinct LLM use case you want to optimize. Each task has:
- A **Scorer** — your code that evaluates response quality (returns 0-1 score + sub-scores)
- **Fixtures** — golden test cases (system_prompt + user_prompt pairs) that exercise representative scenarios
- A **Quality Floor** — minimum acceptable quality (0-1). Models below this are excluded from recommendations

### Pareto Frontier

A model is on the Pareto frontier if no other model beats it on BOTH quality AND cost. The auto-selector picks the cheapest Pareto-optimal model above your quality floor.

### Cost-Per-Success

The effective cost accounts for time wasted on failures:

```
cost_per_success = (mean_cost + cycle_time_cost * (1/quality - 1)) / quality
```

Where `cycle_time_cost` is the USD opportunity cost of a wasted LLM cycle (default: $0.002).

### Auto-Selector Lifecycle

```
[Benchmarks run] → [Results stored] → selector.refresh() → [Pareto analysis]
                                                                    ↓
[Production call] ← selector.get_best_model("task") ← [Recommendation cached]
```

Call `selector.refresh()` periodically (e.g., start of each work cycle) to update recommendations.

## Provider: OpenRouter with BYOK

All LLM calls go through [OpenRouter](https://openrouter.ai/) using the OpenAI-compatible SDK.

**Why OpenRouter?**
- **Actual cost on every response** — `usage.cost` gives real billed USD, not estimates
- **100+ models** — Anthropic, Google, Mistral, OpenAI, Meta, and more
- **Zero markup with BYOK** — first 1M requests/month free when you bring your own API keys
- **Prompt caching** — automatic pass-through for Anthropic, OpenAI, Gemini cache savings
- **Provider routing** — sort by price, latency, or throughput; fallback chains

### Setup

1. Get an [OpenRouter API key](https://openrouter.ai/keys)
2. (Optional) Add your provider keys at [Settings > Integrations](https://openrouter.ai/settings/integrations) for BYOK pricing

### Provider routing options

```python
# Route to cheapest provider for a model
response = client.call("anthropic/claude-haiku-4.5", system, user, provider_sort="price")

# Route to fastest provider
response = client.call("anthropic/claude-haiku-4.5", system, user, provider_sort="latency")

# Fallback chain
response = client.call(
    "anthropic/claude-haiku-4.5", system, user,
    fallback_models=["google/gemini-2.5-flash", "openai/gpt-4.1-mini"],
)
```

## Configuration Reference

### TaskDefinition

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Task identifier (e.g., "summarize") |
| `scorer` | `QualityScorer` | required | Your scoring implementation |
| `fixtures` | `list[Fixture]` | required | Test cases for benchmarking |
| `quality_floor` | `float` | 0.5 | Minimum quality for recommendation (0-1) |
| `models` | `list[str] \| None` | None | Subset of model configs to test; None = all |

### ModelConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Short identifier (e.g., "haiku") |
| `model` | `str` | required | OpenRouter model slug |
| `thinking_budget` | `int` | 0 | Extended thinking token budget; 0 = disabled |

### Built-in Models

21 models ship out of the box:

| Config Name | Model | Provider |
|-------------|-------|----------|
| `haiku` | claude-haiku-4.5 | Anthropic |
| `sonnet` | claude-sonnet-4 | Anthropic |
| `gemini-2.5-flash` | gemini-2.5-flash | Google |
| `gemini-2.5-flash-lite` | gemini-2.5-flash-lite | Google |
| `gemini-3-flash` | gemini-3-flash-preview | Google |
| `devstral-small` | devstral-small-2507 | Mistral |
| `gpt-4.1-mini` | gpt-4.1-mini | OpenAI |
| `gpt-4.1-nano` | gpt-4.1-nano | OpenAI |
| ... | + thinking variants and free models | |

Add custom models:
```python
from llm_bench import ModelConfig, ModelRegistry

registry = ModelRegistry()
registry.add(ModelConfig("my-model", "provider/model-name", thinking_budget=1024))
```

## Dashboard

```bash
pip install llm-bench[dashboard]
```

### Standalone
```bash
llm-bench dashboard --port 8765
```

### Mount in your FastAPI app
```python
from llm_bench.dashboard import create_benchmark_router

router = create_benchmark_router(storage=storage, tasks=tasks)
app.include_router(router, prefix="/benchmarks")
```

### Theming

Override CSS variables to match your app:

```css
:root {
  --llm-bench-bg: #1a1a2e;
  --llm-bench-surface: #16213e;
  --llm-bench-text: #e0e0e0;
  --llm-bench-green: #00b894;
  --llm-bench-red: #e17055;
  --llm-bench-blue: #74b9ff;
  /* ... see benchmark.html for all variables */
}
```

## CLI Reference

```bash
# Setup wizard
llm-bench init
llm-bench init --from config.yaml

# Run benchmarks
llm-bench run
llm-bench run -n 3                    # 3 runs per combo
llm-bench run -c my_config.py

# Analyze results from JSON files
llm-bench analyze results/*.json

# Start dashboard
llm-bench dashboard
llm-bench dashboard --port 9000 --host 0.0.0.0
```

## Storage Backends

### SQLite (default)
```python
from llm_bench import SQLiteStorage
storage = SQLiteStorage()                    # ~/.llm-bench/benchmarks.db
storage = SQLiteStorage("./my-benchmarks.db")  # custom path
```

### JSON Files
```python
from llm_bench import JSONFileStorage
storage = JSONFileStorage()                  # ~/.llm-bench/results/
storage = JSONFileStorage("./results/")      # custom directory
```

### PostgreSQL
```bash
pip install llm-bench[postgres]
```
```python
from llm_bench.storage.postgres import PostgresStorage
storage = PostgresStorage("postgresql://user:pass@host/db")
```

## Continuous Benchmarking

Run one random benchmark every hour in the background:

```python
from llm_bench import ContinuousRunner

continuous = ContinuousRunner(
    runner=runner,
    interval_seconds=3600,
    excluded_configs={"sonnet", "sonnet-think-1k"},  # skip expensive models
    should_pause=lambda: my_spend_limit_reached(),
)
continuous.start()

# Later
continuous.stop()
```

## Cost Tracking

All costs come from OpenRouter's `usage.cost` field — actual billed USD on every response. No pricing tables to maintain, no estimation drift.

## API Reference

### Core Types
- `SimpleFixture(fixture_id, system_prompt, user_prompt)` — basic fixture
- `TaskDefinition(name, scorer, fixtures, quality_floor, models)` — task configuration
- `BenchmarkResult` — result of a single benchmark run
- `ModelConfig(name, model, thinking_budget)` — model configuration

### Client
- `Client(api_key)` — OpenRouter API client
- `client.call(model, system, user_message, ...)` — make an LLM call

### Runner
- `BenchmarkRunner(client, tasks, registry, storage)` — benchmark executor
- `runner.run_one(task, config, fixture)` — single benchmark
- `runner.run_task(task, n_runs)` — all combos for one task
- `runner.run_all(n_runs)` — all combos for all tasks

### Analysis
- `aggregate(results)` — group by (task, config), compute stats
- `pareto_frontier(configs)` — find non-dominated configs
- `cost_per_success(stats)` — effective cost including failure waste
- `recommend(task, configs, quality_floor)` — pick cheapest above floor
- `format_table(task, configs)` — human-readable comparison table

### Auto-Selector
- `AutoSelector(storage, tasks)` — benchmark-driven model selector
- `selector.refresh()` — re-analyze and update recommendations
- `selector.get_best_model(task)` — get optimal model slug for a task
- `selector.get_best_config(task)` — get full ModelConfig

### Storage
- `SQLiteStorage(db_path)` — SQLite backend
- `JSONFileStorage(directory)` — JSON file backend
- `storage.save_result(result)` / `storage.load_results()`

### Dashboard
- `create_benchmark_router(storage, tasks)` — FastAPI router

### Scheduler
- `ContinuousRunner(runner, interval_seconds, ...)` — background benchmarking

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │             Your Application            │
                    │                                         │
                    │  ┌─────────────┐  ┌──────────────────┐  │
                    │  │ Your Scorer │  │  Your Fixtures    │  │
                    │  └──────┬──────┘  └────────┬─────────┘  │
                    │         │                  │            │
                    └─────────┼──────────────────┼────────────┘
                              │                  │
                    ┌─────────▼──────────────────▼────────────┐
                    │              llm-bench                   │
                    │                                         │
                    │  ┌──────────┐  ┌───────────────────┐   │
                    │  │  Runner  │──│   Storage (SQLite) │   │
                    │  └────┬─────┘  └─────────┬─────────┘   │
                    │       │                  │              │
                    │  ┌────▼─────┐  ┌─────────▼──────────┐  │
                    │  │  Client  │  │  AutoSelector       │  │
                    │  └────┬─────┘  │  (Pareto analysis)  │  │
                    │       │        └─────────┬──────────┘  │
                    └───────┼──────────────────┼─────────────┘
                            │                  │
                            ▼                  ▼
                    ┌───────────────┐  ┌───────────────────┐
                    │  OpenRouter   │  │  get_best_model()  │
                    │  (100+ models)│  │  → production call │
                    └───────────────┘  └───────────────────┘
```

## License

MIT
