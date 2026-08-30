"""Offline check that the harness itself works.

Swaps the embedding model for a deterministic lexical stub, so the whole path
- load PDFs, chunk, index, retrieve, score, keep/discard, log - runs with no
network and no GPU. It proves the machinery is sound. It says nothing about
how good the real embeddings are; that needs `python eval/loop.py --sweep`.

Usage:
    python eval/smoke_test.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from langchain_core.embeddings import Embeddings  # noqa: E402

import rag_core  # noqa: E402
from rag_core import normalize  # noqa: E402

DIMENSIONS = 512

# Model names the stub will pretend to know. Anything else raises, the way the
# real loader does on a bad repo id - otherwise the error path below is fake.
KNOWN_MODELS = {
    "thenlper/gte-small",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
}


class StubEmbeddings(Embeddings):
    """Hashed bag-of-words. Crude, deterministic, and offline.

    Real enough that lexically similar text lands near each other, so the
    retrieval path is genuinely exercised rather than fed random vectors.
    """

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * DIMENSIONS
        for token in normalize(text).split():
            digest = hashlib.md5(token.encode()).digest()
            vec[int.from_bytes(digest[:4], "big") % DIMENSIONS] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def main() -> int:
    from eval.config import CONFIG
    from eval.evaluate import evaluate, load_benchmark
    from eval import loop

    def stub_embeddings(config):
        if config["embedding_model"] not in KNOWN_MODELS:
            raise OSError(f"model not found: {config['embedding_model']}")
        return StubEmbeddings()

    rag_core.get_embeddings = stub_embeddings
    tmp = Path(tempfile.mkdtemp(prefix="rag_smoke_"))
    rag_core.INDEX_CACHE = tmp
    loop.RESULTS_PATH = tmp / "results.jsonl"

    failures = []

    def check(label, condition, detail=""):
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
        if not condition:
            failures.append(label)

    try:
        print("\n1. corpus and benchmark")
        chunks = rag_core.load_chunks(CONFIG)
        benchmark = load_benchmark()
        check("PDFs chunked", len(chunks) > 20, f"{len(chunks)} chunks")
        check("benchmark loaded", len(benchmark) > 20, f"{len(benchmark)} questions")

        print("\n2. index build and cache")
        store = rag_core.build_or_load_index(CONFIG)
        fingerprint = rag_core.index_fingerprint(CONFIG)
        check("index built", store is not None)
        check("index cached to disk", (tmp / f"idx_{fingerprint}").exists())
        check("fingerprint tracks index fields",
              rag_core.index_fingerprint({**CONFIG, "chunk_size": 999}) != fingerprint)
        check("fingerprint ignores query-time fields",
              rag_core.index_fingerprint({**CONFIG, "k": 99}) == fingerprint)

        print("\n3. retrieval and citation-ready context")
        docs = rag_core.get_retriever(CONFIG).invoke("How long do I have to return an item?")
        context = rag_core.format_context(docs)
        check("retriever honours k", len(docs) == CONFIG["k"], f"{len(docs)} docs")
        check("context carries source labels", context.count("[source:") == len(docs))
        check("labels name a real PDF and page", ".pdf p." in context)

        print("\n4. scoring")
        result = evaluate(CONFIG, mode="retrieval", benchmark=benchmark)
        metrics = result["metrics"]
        check("all answerable questions scored",
              len(result["per_item"]) == metrics["n_answerable"])
        check("metrics in range",
              all(0.0 <= metrics[m] <= 1.0 for m in ("fact_recall", "source_recall", "mrr")),
              f"fact_recall={metrics['fact_recall']} mrr={metrics['mrr']}")
        check("stub retrieves something real", metrics["fact_recall"] > 0.0,
              "a zero here would mean the matcher is broken, not the model")

        print("\n5. scoring is deterministic")
        again = evaluate(CONFIG, mode="retrieval", benchmark=benchmark)
        check("same config scores identically",
              again["metrics"]["fact_recall"] == metrics["fact_recall"])

        print("\n6. k affects context size as expected")
        small = evaluate({**CONFIG, "k": 2}, mode="retrieval", benchmark=benchmark)
        check("smaller k means less context",
              small["metrics"]["context_chars"] < metrics["context_chars"],
              f"k=2 {small['metrics']['context_chars']:.0f} vs "
              f"k={CONFIG['k']} {metrics['context_chars']:.0f} chars")

        print("\n7. keep/discard loop")
        best = loop.run_sweep(dict(CONFIG), [{"k": 2}, {"k": 4}], mode="retrieval")
        logged = [line for line in loop.RESULTS_PATH.read_text().splitlines() if line.strip()]
        check("every experiment logged", len(logged) == 3, f"{len(logged)} rows (baseline + 2)")
        check("decisions recorded",
              all(any(d in row for d in ("kept", "discarded", "baseline")) for row in logged))
        check("loop returns a config", "config" in best)

        print("\n8. bad candidates do not kill the run")
        loop.run_sweep(dict(CONFIG), [{"embedding_model": "does/not-exist"}], mode="retrieval")
        check("error logged, run continued",
              any("error" in row for row in loop.RESULTS_PATH.read_text().splitlines()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All harness checks passed (stub embeddings, no network).")
    print("Real numbers: python eval/loop.py --sweep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
