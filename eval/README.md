# Retrieval eval harness

An eval loop for this chatbot's retrieval pipeline, built on the pattern from
[karpathy/autoresearch](https://github.com/karpathy/autoresearch): edit one
file, score against a fixed benchmark, keep the change only if a metric moves,
log every attempt.

The scoring half of this problem is well served by
[RAGAS](https://github.com/explodinggradients/ragas), promptfoo and DeepEval.
What they don't provide is the closed keep/discard loop, which is the part
borrowed here.

## Quick start

```bash
python eval/smoke_test.py        # offline check that the harness works
python eval/evaluate.py          # score the current config
python eval/loop.py --sweep      # run the candidate sweep, keep/discard, log
```

`smoke_test.py` needs no network. The other two download an embedding model on
first run, and `--mode full` additionally needs `ollama serve` running with
`gemma3:1b` pulled.

## Files

| File | Role |
|---|---|
| `config.py` | The only file an experiment may change. `BASELINE` is the shipped starting point and is never edited. |
| `benchmark.yaml` | 28 answerable questions + 3 unanswerable ones, with the source document and acceptable answer keys. |
| `evaluate.py` | Scores one config. `--mode retrieval` (deterministic) or `--mode full` (adds the LLM). |
| `loop.py` | Runs candidates, keeps or discards each, appends to `results.jsonl`. |
| `smoke_test.py` | Exercises the whole path offline with a stub embedder. |
| `AGENT.md` | Instructions for an agent driving the loop. |
| `results.jsonl` | Append-only experiment log. |

The app and the harness both import `rag_core.py`, so the benchmark scores the
same retrieval code path users talk to, and a config that wins here ships to
`app.py` with nothing to copy across.

## Metrics

`fact_recall` is primary: the share of questions whose answer text reached the
context window. It is deterministic, which is what makes keep/discard
trustworthy — an LLM-judged metric moves on its own between runs and would have
you keeping noise.

`context_chars` is the tie-breaker. Two configs that answer equally well are
not equal; the one using half the context is cheaper and faster.

`--mode full` adds `answer_accuracy`, `citation_rate`, `abstain_accuracy` and
`mean_latency_s`.

## Limitations

28 questions over 4 documents is a regression test, not a generalization
benchmark. It will catch a change that makes retrieval worse. It will not tell
you the system is good, and a config tuned against it is tuned against these 28
questions — expect some of the gain to be overfitting.

The answer keys are substring matches after normalization. They confirm the
right text was retrieved, not that the model reasoned about it correctly.
