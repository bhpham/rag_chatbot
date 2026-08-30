"""Score one configuration against the benchmark.

Two modes:

  retrieval  (default) — deterministic, no LLM. Measures whether the evidence
             needed to answer reached the context window at all. Fast enough to
             sweep dozens of configs, and repeatable, which is what makes
             keep/discard decisions trustworthy.

  full       — adds generation. Measures what the user actually sees, including
             whether the model cites sources and whether it declines questions
             the corpus can't answer. Slower and non-deterministic; used to
             confirm a retrieval win survives end to end.

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --mode full
    python eval/evaluate.py --set k=4 --set chunk_size=500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

import rag_core  # noqa: E402
from rag_core import normalize, source_label  # noqa: E402

BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmark.yaml"
ABSTAIN_MARKER = "i don't know based on the retrieved documents"


def load_benchmark(path: Path = BENCHMARK_PATH):
    return yaml.safe_load(path.read_text())


def _first_hit_rank(docs, keys) -> int | None:
    """1-indexed rank of the first chunk containing any answer key."""
    normalized_keys = [normalize(k) for k in keys]
    for rank, doc in enumerate(docs, start=1):
        haystack = normalize(doc.page_content)
        if any(k in haystack for k in normalized_keys):
            return rank
    return None


def evaluate(config: dict, mode: str = "retrieval", rebuild: bool = False,
             benchmark=None, verbose: bool = False) -> dict:
    benchmark = benchmark if benchmark is not None else load_benchmark()
    answerable = [q for q in benchmark if not q.get("abstain")]
    abstainable = [q for q in benchmark if q.get("abstain")]

    retriever = rag_core.get_retriever(config, rebuild=rebuild)
    llm = prompt = None
    if mode == "full":
        from langchain_core.prompts import ChatPromptTemplate

        llm = rag_core.get_llm(config)
        prompt = ChatPromptTemplate.from_template(config["system_template"])

    fact_hits = source_hits = 0
    reciprocal_ranks: list[float] = []
    context_sizes: list[int] = []
    answer_hits = citation_hits = 0
    latencies: list[float] = []
    per_item = []

    for item in answerable:
        started = time.time()
        docs = retriever.invoke(item["question"])
        context = rag_core.format_context(docs)
        context_sizes.append(len(context))

        rank = _first_hit_rank(docs, item["answer_keys"])
        fact_hit = rank is not None
        fact_hits += fact_hit
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        source_hit = any(
            item["source"] in source_label(d) for d in docs
        )
        source_hits += source_hit

        record = {
            "id": item["id"],
            "fact_in_context": fact_hit,
            "gold_source_retrieved": source_hit,
            "first_hit_rank": rank,
        }

        if mode == "full":
            messages = prompt.format_messages(context=context, question=item["question"])
            text = llm.invoke(messages)
            latencies.append(time.time() - started)
            normalized = normalize(text)
            answer_hit = any(normalize(k) in normalized for k in item["answer_keys"])
            answer_hits += answer_hit
            citation_hits += "[source:" in text.lower()
            record.update(answer_correct=answer_hit, cited="[source:" in text.lower())

        per_item.append(record)
        if verbose:
            print(f"  {record['id']:22s} {'HIT ' if fact_hit else 'MISS'} rank={rank}")

    n = len(answerable)
    metrics = {
        "fact_recall": round(fact_hits / n, 4),
        "source_recall": round(source_hits / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "context_chars": round(sum(context_sizes) / n, 1),
        "n_answerable": n,
    }

    if mode == "full":
        correct_abstentions = 0
        for item in abstainable:
            result = rag_core.answer(
                item["question"], config, retriever=retriever, llm=llm, prompt=prompt
            )
            correct_abstentions += ABSTAIN_MARKER in normalize(result["answer"])
        metrics.update(
            answer_accuracy=round(answer_hits / n, 4),
            citation_rate=round(citation_hits / n, 4),
            abstain_accuracy=round(correct_abstentions / len(abstainable), 4)
            if abstainable else None,
            mean_latency_s=round(sum(latencies) / len(latencies), 2) if latencies else None,
        )

    return {"mode": mode, "metrics": metrics, "per_item": per_item}


def primary_metric(mode: str) -> str:
    """The number keep/discard is decided on."""
    return "answer_accuracy" if mode == "full" else "fact_recall"


def _apply_overrides(config: dict, overrides: list[str]) -> dict:
    config = dict(config)
    for override in overrides:
        key, _, raw = override.partition("=")
        if key not in config:
            raise SystemExit(f"unknown config key: {key}")
        current = config[key]
        if isinstance(current, bool):
            config[key] = raw.lower() in {"1", "true", "yes"}
        elif isinstance(current, int):
            config[key] = int(raw)
        elif isinstance(current, float):
            config[key] = float(raw)
        else:
            config[key] = raw
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a CONFIG field for this run")
    parser.add_argument("--baseline", action="store_true",
                        help="score the shipped baseline instead of current CONFIG")
    parser.add_argument("--rebuild", action="store_true", help="force a FAISS rebuild")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--verbose", action="store_true", help="print per-question results")
    args = parser.parse_args()

    from eval.config import BASELINE, CONFIG

    config = dict(CONFIG)
    if args.baseline:
        config.update(BASELINE)
    config = _apply_overrides(config, args.overrides)

    result = evaluate(config, mode=args.mode, rebuild=args.rebuild, verbose=args.verbose)

    if args.json:
        print(json.dumps({"config": {k: v for k, v in config.items()
                                     if k != "system_template"},
                          **result}, indent=2))
        return

    key = primary_metric(args.mode)
    print(f"\nmode={args.mode}  k={config['k']}  chunk={config['chunk_size']}/"
          f"{config['chunk_overlap']}  emb={config['embedding_model']}")
    print("-" * 62)
    for name, value in result["metrics"].items():
        marker = "  <-- primary" if name == key else ""
        print(f"  {name:18s} {value}{marker}")
    print()


if __name__ == "__main__":
    main()
