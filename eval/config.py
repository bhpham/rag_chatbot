"""The only file an experiment may change.

Mirrors the constraint from karpathy/autoresearch: the agent edits one file,
everything else is fixed. That keeps every experiment a small, reviewable diff
and stops the harness from being tuned to flatter a result.

Edit CONFIG, run `python eval/evaluate.py`, keep the change only if the primary
metric improves. `eval/loop.py` automates exactly that.
"""

CONFIG = {
    # --- indexing (changing any of these forces a FAISS rebuild) ---
    "chunk_size": 300,
    "chunk_overlap": 30,
    "embedding_model": "thenlper/gte-small",
    # --- retrieval ---
    "k": 8,
    "search_type": "similarity",  # "similarity" | "mmr"
    "mmr_lambda": 0.5,            # only used when search_type == "mmr"
    # --- generation (only used by --mode full) ---
    "llm_model": "gemma3:1b",
    "temperature": 0.1,
    "system_template": """
You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.

Rules:
1) Use ONLY the provided CONTEXT to answer.
2) If the answer is not in the context, say exactly: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
4) Cite the source of every fact as [source: <filename>] using the source label
   shown above each context block.

CONTEXT:
{context}

USER:
{question}
""",
}

# Baseline = the settings the app shipped with, before any experiment ran.
# Never edit this. It is the reference point every result is measured against.
BASELINE = {
    "chunk_size": 300,
    "chunk_overlap": 30,
    "embedding_model": "thenlper/gte-small",
    "k": 8,
    "search_type": "similarity",
    "mmr_lambda": 0.5,
    "llm_model": "gemma3:1b",
    "temperature": 0.1,
}
