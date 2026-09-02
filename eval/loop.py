"""The keep/discard loop.

The autoresearch pattern, applied to retrieval: propose a change, score it
against a fixed benchmark, keep it only if the primary metric improves by more
than the noise floor, log every attempt either way.

Two ways to drive it:

  --sweep     run the built-in candidate list unattended (no agent needed).
  --candidates candidates.json
              run a list you or an agent wrote.

Every experiment is appended to eval/results.jsonl, kept or discarded. The
discarded ones are the point: they are the evidence that a setting you were
about to ship was not actually better.

Usage:
    python eval/loop.py --sweep
    python eval/loop.py --sweep --mode full
    python eval/loop.py --sweep --apply-best
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.evaluate import evaluate, load_benchmark, primary_metric  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "results.jsonl"

# Minimum improvement that counts as real. With 28 answerable questions one
# question is worth ~0.036, so anything smaller than a whole question moving is
# not a result, it is rounding.
MIN_DELTA = 0.01

# Candidates worth trying first, ordered roughly by how much they usually move
# retrieval. Each is a partial config, merged over the current best.
SWEEP_CANDIDATES = [
    {"k": 4},
    {"k": 6},
    {"k": 12},
    {"chunk_size": 500, "chunk_overlap": 50},
    {"chunk_size": 800, "chunk_overlap": 80},
    {"chunk_size": 1000, "chunk_overlap": 150},
    {"chunk_size": 500, "chunk_overlap": 150},
    {"embedding_model": "BAAI/bge-small-en-v1.5"},
    {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"search_type": "mmr", "mmr_lambda": 0.5},
]


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def log_experiment(record: dict) -> None:
    with RESULTS_PATH.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def describe(config: dict, previous: dict | None = None) -> str:
    """Show only what this candidate changed, not the whole config."""
    if previous is None:
        return "baseline"
    changed = {k: v for k, v in config.items()
               if k != "system_template" and previous.get(k) != v}
    return ", ".join(f"{k}={v}" for k, v in changed.items()) or "no change"


def run_sweep(base_config: dict, candidates: list[dict], mode: str,
              greedy: bool = True) -> dict:
    benchmark = load_benchmark()
    key = primary_metric(mode)

    print(f"\nbenchmark: {len(benchmark)} questions  |  primary metric: {key}")
    print("=" * 78)

    started = time.time()
    result = evaluate(base_config, mode=mode, benchmark=benchmark)
    best_score = result["metrics"][key]
    best_context = result["metrics"]["context_chars"]
    best_config = dict(base_config)

    print(f"baseline           {key}={best_score:.4f}  "
          f"context={best_context:.0f} chars  ({time.time() - started:.1f}s)")
    log_experiment({
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(), "mode": mode, "label": "baseline",
        "config": {k: v for k, v in base_config.items() if k != "system_template"},
        "metrics": result["metrics"], "decision": "baseline",
    })

    for candidate in candidates:
        # Greedy: build on the current best. Otherwise vary the baseline only,
        # which isolates each change but can't find combinations.
        trial = copy.deepcopy(best_config if greedy else base_config)
        trial.update(candidate)
        label = describe(trial, best_config if greedy else base_config)

        started = time.time()
        try:
            result = evaluate(trial, mode=mode, benchmark=benchmark)
        except Exception as exc:  # a bad candidate must not kill the run
            print(f"  {label:44s} ERROR {type(exc).__name__}: {exc}")
            log_experiment({
                "ts": datetime.now(timezone.utc).isoformat(), "git_sha": git_sha(),
                "mode": mode, "label": label,
                "config": {k: v for k, v in trial.items() if k != "system_template"},
                "error": f"{type(exc).__name__}: {exc}", "decision": "error",
            })
            continue

        score = result["metrics"][key]
        context = result["metrics"]["context_chars"]
        delta = score - best_score

        # Keep on a real improvement. Also keep an exact tie that cuts context,
        # since the same answers on less context is cheaper and faster.
        improved = delta > MIN_DELTA
        cheaper_tie = abs(delta) <= MIN_DELTA and context < best_context * 0.9
        keep = improved or cheaper_tie

        decision = "kept" if keep else "discarded"
        reason = ("improved" if improved else
                  "same score, less context" if cheaper_tie else "no improvement")

        print(f"  {label:44s} {key}={score:.4f} ({delta:+.4f}) "
              f"context={context:.0f}  {decision.upper()} [{reason}]  "
              f"({time.time() - started:.1f}s)")

        log_experiment({
            "ts": datetime.now(timezone.utc).isoformat(), "git_sha": git_sha(),
            "mode": mode, "label": label,
            "config": {k: v for k, v in trial.items() if k != "system_template"},
            "metrics": result["metrics"], "delta": round(delta, 4),
            "decision": decision, "reason": reason,
        })

        if keep:
            best_score, best_context, best_config = score, context, trial

    print("=" * 78)
    print(f"best {key}={best_score:.4f}  context={best_context:.0f} chars")
    print(f"winning config: {describe(best_config, base_config) or 'baseline unchanged'}")
    print(f"\nfull log: {RESULTS_PATH}")
    return {"config": best_config, key: best_score, "context_chars": best_context}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="run built-in candidates")
    parser.add_argument("--candidates", type=Path,
                        help="JSON file with a list of partial configs")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--isolate", action="store_true",
                        help="test each candidate against the baseline, not the best-so-far")
    parser.add_argument("--apply-best", action="store_true",
                        help="write the winning values back into eval/config.py")
    args = parser.parse_args()

    from eval.config import CONFIG

    if args.candidates:
        candidates = json.loads(args.candidates.read_text())
    elif args.sweep:
        candidates = SWEEP_CANDIDATES
    else:
        parser.error("pass --sweep or --candidates")

    best = run_sweep(dict(CONFIG), candidates, args.mode, greedy=not args.isolate)

    if args.apply_best:
        apply_best(best["config"])
        print("\nwrote winning values into eval/config.py")


def apply_best(config: dict) -> None:
    """Patch the scalar fields of CONFIG in place, leaving the prompt alone."""
    import re

    path = Path(__file__).resolve().parent / "config.py"
    text = path.read_text()
    head, marker, tail = text.partition("# Baseline =")
    for field in ("chunk_size", "chunk_overlap", "embedding_model", "k",
                  "search_type", "mmr_lambda", "temperature"):
        value = config[field]
        literal = f'"{value}"' if isinstance(value, str) else repr(value)
        head = re.sub(rf'^(\s*"{field}":\s*).*?,(\s*(?:#.*)?)$',
                      lambda m: f"{m.group(1)}{literal},{m.group(2)}",
                      head, count=1, flags=re.MULTILINE)
    path.write_text(head + marker + tail)


if __name__ == "__main__":
    main()
