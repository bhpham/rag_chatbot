# rag_chatbot

A customer support chatbot over four synthetic policy PDFs, and — the more
interesting half — the evaluation harness that measures it.

The chatbot is ordinary: PDFs → chunks → FAISS → top-k retrieval → local LLM.
The harness is the part worth reading. It scores a configuration against a fixed
benchmark, keeps a change only when it beats what's already there, and logs
every experiment it throws away.

## The finding

The shipped settings retrieved the top 8 chunks per question. Cutting that to 4
changed nothing except cost:

| config | answer recall | mean reciprocal rank | context per question |
|---|---|---|---|
| k=8 (original) | 100% | 0.9405 | 2,706 chars |
| **k=4** | **100%** | **0.9405** | **1,339 chars** |
| k=6 | 100% | 0.9405 | 2,028 chars |
| k=12 | 100% | 0.9405 | 3,994 chars |

Every answer was already in the top 4 results, so chunks 5-8 were padding on
every question — half the context for identical evidence.

The other result is about the benchmark itself: it scored **100% at baseline**,
so ten experiments produced zero improvements. A benchmark that starts at
ceiling can only tell you when you've made things worse. Making it harder is the
next job, not tuning against it.

Full log, kept and discarded alike: [`eval/results.jsonl`](eval/results.jsonl).

## Quickstart

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
ollama pull gemma3:1b          # only needed for the app and --mode full
```

Run the chatbot:

```bash
streamlit run app.py
```

Run the eval:

```bash
python3 eval/smoke_test.py       # offline check that the harness works, ~10s
python3 eval/evaluate.py         # score the current config
python3 eval/loop.py --sweep     # run candidates, keep/discard, log
```

`smoke_test.py` needs no network and no LLM. `evaluate.py` and `loop.py`
download an embedding model on first run. `--mode full` additionally needs
`ollama serve` running.

## The benchmark

[`eval/benchmark.yaml`](eval/benchmark.yaml) holds 31 questions drawn from the
policy documents:

- **28 answerable**, each with its source document and a list of acceptable
  answer keys. Every key was verified to appear in its source document, so a
  zero score means the retrieval failed, not the benchmark.
- **3 unanswerable**, to measure whether the model declines or invents an
  answer.

## Metrics

`fact_recall` is primary: the share of questions whose answer text reached the
context window. It is deterministic — no LLM in the decision path — which is
what makes keep/discard trustworthy. An LLM-judged metric drifts between
identical runs and would have you keeping noise.

`context_chars` is the tie-breaker. Two configs that answer equally well are not
equal; the one using half the context is cheaper.

`--mode full` adds `answer_accuracy`, `citation_rate`, `abstain_accuracy` and
`mean_latency_s`. It calls the LLM, so it is slower and not perfectly
repeatable.

## What this does not measure

- **Whether the answers are good.** The numbers above confirm the right passage
  reaches the model. They say nothing about what it writes with it.
- **Generalization.** 28 questions over four short documents is a regression
  test. It catches changes that make retrieval worse; it does not prove the
  system is good, and a config tuned against it is tuned against these 28
  questions.
- **Paraphrase.** Answer keys are substring matches after normalization, so a
  correct answer worded differently scores as a miss.

## Layout

| Path | Role |
|---|---|
| `app.py` | Streamlit chat UI |
| `rag_core.py` | Pipeline shared by the app and the harness, so the benchmark scores the same code path users talk to |
| `eval/config.py` | The only file an experiment may change; `BASELINE` records the shipped settings |
| `eval/benchmark.yaml` | The questions |
| `eval/evaluate.py` | Scores one configuration |
| `eval/loop.py` | Runs candidates, keeps or discards, appends to `results.jsonl` |
| `eval/smoke_test.py` | Offline check of the harness itself |
| `eval/AGENT.md` | Instructions for an agent driving the loop |

## Credit

The keep-or-revert loop is adapted from
[karpathy/autoresearch](https://github.com/karpathy/autoresearch), where an
agent edits a single file, runs a fixed-budget experiment, and keeps the change
only if a metric improves. Built for GPT pretraining; the control loop
generalizes.
