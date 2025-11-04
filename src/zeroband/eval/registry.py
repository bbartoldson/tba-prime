from typing import Literal

from datasets import Dataset, load_dataset, load_from_disk

Benchmark = Literal["math500", "aime24", "aime25", "livecodebench-v5", "countdown"]

_BENCHMARKS_DATASET_NAMES: dict[Benchmark, str] = {
    "math500": "PrimeIntellect/MATH-500",
    "aime24": "PrimeIntellect/AIME-24",
    "aime25": "PrimeIntellect/AIME-25",
    "livecodebench-v5": "PrimeIntellect/LiveCodeBench-v5",
    "countdown": "/p/vast1/bartolds/tba-prime/datasets/countdown/test",
}

_BENCHMARK_DISPLAY_NAMES: dict[Benchmark, str] = {
    "math500": "MATH-500",
    "aime24": "AIME-24",
    "aime25": "AIME-25",
    "livecodebench-v5": "LiveCodeBench-V5",
    "countdown": "Countdown",
}


def get_benchmark_dataset(name: Benchmark) -> Dataset:
    if name == "countdown":
        return load_from_disk(_BENCHMARKS_DATASET_NAMES[name])
    return load_dataset(_BENCHMARKS_DATASET_NAMES[name], split="train")


def get_benchmark_display_name(name: Benchmark) -> str:
    return _BENCHMARK_DISPLAY_NAMES[name]
