#!/usr/bin/env python3
"""Quickstart example — benchmark models on a simple summarization task.

Prerequisites:
    pip install llm-bench
    export OPENROUTER_API_KEY=sk-or-v1-your-key-here

Usage:
    python examples/quickstart.py
"""

import os

import llm_bench


# --- 1. Define a Quality Scorer ---
class SummarizationScorer:
    """Scores summaries on key point coverage and conciseness."""

    def score(self, response: str, fixture, **kwargs) -> dict:
        text = response.lower()

        # Check for key points mentioned in the fixture
        key_points = getattr(fixture, "key_points", ["climate", "2030", "emissions"])
        hits = sum(1 for kp in key_points if kp in text)
        coverage = hits / len(key_points) if key_points else 0.0

        # Check conciseness (under 200 words)
        word_count = len(response.split())
        is_concise = 1.0 if word_count < 200 else max(0.0, 1.0 - (word_count - 200) / 200)

        # Composite score
        score = coverage * 0.7 + is_concise * 0.3

        return {
            "score": score,
            "key_point_coverage": coverage,
            "conciseness": is_concise,
            "word_count": float(word_count),
        }


# --- 2. Define Test Fixtures ---
fixtures = [
    llm_bench.SimpleFixture(
        fixture_id="climate_article",
        system_prompt="Summarize the following article in 2-3 sentences. Be concise and accurate.",
        user_prompt=(
            "The United Nations Climate Change Conference (COP35) concluded with "
            "195 nations agreeing to cut greenhouse gas emissions by 45% below 2020 "
            "levels by 2030. The agreement includes a $500 billion annual climate "
            "finance package for developing nations. Scientists praised the targets "
            "as achievable but warned that implementation must begin immediately to "
            "avoid catastrophic warming beyond 1.5 degrees Celsius."
        ),
    ),
    llm_bench.SimpleFixture(
        fixture_id="tech_article",
        system_prompt="Summarize the following article in 2-3 sentences. Be concise and accurate.",
        user_prompt=(
            "Researchers at MIT have developed a new quantum error correction method "
            "that reduces logical error rates by a factor of 100. The breakthrough "
            "uses topological codes on a 72-qubit processor, achieving below-threshold "
            "performance for the first time. This is considered a critical milestone "
            "toward practical fault-tolerant quantum computing."
        ),
    ),
]

# --- 3. Create Task Definition ---
task = llm_bench.TaskDefinition(
    name="summarize",
    scorer=SummarizationScorer(),
    fixtures=fixtures,
    quality_floor=0.5,
    # Test a subset of cheap models for the quickstart
    models=["haiku", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gpt-4.1-mini", "gpt-4.1-nano"],
)

# --- 4. Set Up ---
api_key = os.environ.get("OPENROUTER_API_KEY", "")
if not api_key:
    print("Error: Set OPENROUTER_API_KEY environment variable")
    print("Get your key at https://openrouter.ai/keys")
    exit(1)

client = llm_bench.Client(api_key=api_key)
storage = llm_bench.SQLiteStorage()  # Stores in ~/.llm-bench/benchmarks.db
runner = llm_bench.BenchmarkRunner(
    client=client, tasks=[task], storage=storage
)

# --- 5. Run Benchmarks ---
print("Running benchmarks...")
print(f"Testing {len(task.models)} models x {len(fixtures)} fixtures = {len(task.models) * len(fixtures)} runs\n")

results = runner.run_all(n_runs=1)

print(f"\nCompleted {len(results)} benchmark runs")

# --- 6. Get Recommendations ---
selector = llm_bench.AutoSelector(storage=storage, tasks=[task])
recs = selector.refresh()

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
for task_name, config_name in recs.items():
    mc = selector.get_best_config(task_name)
    print(f"  {task_name}: {config_name} ({mc.model})")

# --- 7. Print Detailed Table ---
stats = selector.get_stats()
for task_name, configs in stats.items():
    print(llm_bench.format_table(task_name, configs, quality_floor=task.quality_floor))

print("\n\nTo view the dashboard:")
print("  llm-bench dashboard")
print("  Open http://127.0.0.1:8765")
