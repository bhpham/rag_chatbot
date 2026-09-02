# Research program

You are improving the retrieval quality of this RAG chatbot. Work autonomously.

## The one rule

You may edit **`eval/config.py` only**. Not the benchmark, not the scorer, not
the loop, not `rag_core.py`. Changing the yardstick to make a number go up is
the one way to make this whole exercise worthless.

## The loop

1. Read `eval/results.jsonl` first. Do not re-run a configuration that is
   already logged.
2. Form a hypothesis and say it out loud before testing it. "Chunks of 300
   characters split the shipping table across chunks, so table lookups miss"
   is a hypothesis. "Try k=6" is a guess.
3. Edit `CONFIG` in `eval/config.py`.
4. Run `python eval/evaluate.py`.
5. Keep the change if `fact_recall` improves by more than 0.01. Otherwise
   revert it (`git checkout eval/config.py`).
6. Record what happened, including the failures. A discarded experiment is a
   result: it is evidence that a setting you might have shipped is not better.

`python eval/loop.py --sweep` runs steps 3-6 automatically over a candidate
list. Use it for the obvious sweeps; do the hypothesis-driven work yourself.

## What you can change, roughly in order of expected effect

- `chunk_size` / `chunk_overlap` — the corpus has tables (size charts, shipping
  rates, refund timelines). 300 characters is small enough to cut rows apart.
- `k` — more context is not free. Watch `context_chars` alongside recall.
- `embedding_model` — any sentence-transformers model works. Bigger is slower.
- `search_type` — `mmr` trades some relevance for diversity, which may help
  questions whose answer spans two documents.
- `system_template` — only affects `--mode full`.

## Metrics

Primary: **`fact_recall`** — the share of questions where the answer text
actually reached the context window. Deterministic, so a difference between
two runs is a real difference and not sampling noise.

Secondary, and never optimized directly:
- `source_recall` — did the right document show up at all. If this is high
  while `fact_recall` is low, retrieval is finding the right file and the wrong
  passage: a chunking problem, not an embedding problem.
- `mrr` — how near the top the answer landed. Improves when `fact_recall` is
  already saturated.
- `context_chars` — cost and latency proxy. Two configs that score the same are
  not equal; the cheaper one is better.

`--mode full` adds `answer_accuracy`, `citation_rate`, `abstain_accuracy` and
`mean_latency_s`. It calls the LLM, so it is slow and not perfectly repeatable.
Use it to confirm a retrieval win survives generation, not to drive the search.

## Honesty rules

- 28 answerable questions means one question is worth ~0.036. Do not report a
  0.01 move as a finding.
- If a change helps `fact_recall` but hurts `abstain_accuracy`, say so. A bot
  that answers everything confidently is worse than one that declines.
- If nothing beats the baseline, that is the result. Report it. Do not keep
  widening the search until something wins by chance.
