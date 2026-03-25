"""Storage backends for benchmark results."""

from llm_bench.storage.json_file import JSONFileStorage
from llm_bench.storage.sqlite import SQLiteStorage

__all__ = ["JSONFileStorage", "SQLiteStorage"]
