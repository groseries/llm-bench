"""CLI entry point — python -m llm_bench or llm-bench command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_init(args: argparse.Namespace) -> None:
    """Run the setup wizard."""
    from llm_bench.setup_wizard import run_from_config, run_wizard

    if args.config:
        run_from_config(args.config)
    else:
        run_wizard(output_dir=args.output)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze benchmark results from JSON files."""
    from llm_bench.analyze import (
        aggregate,
        format_table,
        get_recommendations,
    )

    results: list[dict] = []
    for path_str in args.files:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not read {path}: {e}")

    if not results:
        print("No results loaded.")
        sys.exit(1)

    print(f"Loaded {len(results)} benchmark runs")

    agg = aggregate(results)
    for task_name, configs in sorted(agg.items()):
        print(format_table(task_name, configs))

    recs = get_recommendations(results)
    if recs:
        print(f"\nRecommendations: {recs}")


def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmarks using the project config."""
    config_path = Path(args.config or "llm_bench_config.py")
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Run 'llm-bench init' first to create a project scaffold.")
        sys.exit(1)

    # Import the config module
    import importlib.util

    spec = importlib.util.spec_from_file_location("llm_bench_config", config_path)
    if not spec or not spec.loader:
        print(f"Could not load config: {config_path}")
        sys.exit(1)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    runner = getattr(config, "runner", None)
    if not runner:
        print("Config must export a 'runner' (BenchmarkRunner instance)")
        sys.exit(1)

    n_runs = args.n_runs or 1
    print(f"Running benchmarks ({n_runs} runs per combo)...")
    results = runner.run_all(n_runs=n_runs)
    print(f"Completed {len(results)} benchmark runs")

    # Print recommendations
    selector = getattr(config, "selector", None)
    if selector:
        recs = selector.refresh()
        if recs:
            print(f"\nRecommendations: {recs}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Start the dashboard web server."""
    try:
        import uvicorn
        from fastapi import FastAPI
    except ImportError:
        print("Dashboard requires FastAPI. Install with: pip install llm-bench[dashboard]")
        sys.exit(1)

    from llm_bench.dashboard import create_benchmark_router

    # Try to load config
    config_path = Path(args.config or "llm_bench_config.py")
    if config_path.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("llm_bench_config", config_path)
        if spec and spec.loader:
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            storage = getattr(config, "storage", None)
            tasks = getattr(config, "tasks", [])
        else:
            storage = None
            tasks = []
    else:
        # Use default storage with no tasks
        from llm_bench.storage import SQLiteStorage

        storage = SQLiteStorage()
        tasks = []

    if not storage:
        from llm_bench.storage import SQLiteStorage

        storage = SQLiteStorage()

    app = FastAPI(title="llm-bench Dashboard")
    router = create_benchmark_router(storage=storage, tasks=tasks)
    app.include_router(router)

    host = args.host or "127.0.0.1"
    port = args.port or 8765
    print(f"Starting dashboard at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-bench",
        description="Benchmark LLM models on quality-per-cost and auto-select the best model per task.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # init
    init_parser = subparsers.add_parser("init", help="Set up a new project with the setup wizard")
    init_parser.add_argument("--from", dest="config", help="Config file (YAML/TOML/JSON) for non-interactive setup")
    init_parser.add_argument("-o", "--output", help="Output directory", default=".")
    init_parser.set_defaults(func=cmd_init)

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze benchmark results from JSON files")
    analyze_parser.add_argument("files", nargs="+", help="JSON result files to analyze")
    analyze_parser.set_defaults(func=cmd_analyze)

    # run
    run_parser = subparsers.add_parser("run", help="Run benchmarks using project config")
    run_parser.add_argument("-c", "--config", help="Config file path", default="llm_bench_config.py")
    run_parser.add_argument("-n", "--n-runs", type=int, default=1, help="Runs per model/fixture combo")
    run_parser.set_defaults(func=cmd_run)

    # dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Start the web dashboard")
    dash_parser.add_argument("-c", "--config", help="Config file path", default="llm_bench_config.py")
    dash_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    dash_parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    dash_parser.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
