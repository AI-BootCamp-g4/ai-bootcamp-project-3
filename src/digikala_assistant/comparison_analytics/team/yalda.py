"""Semantic review retrieval with the same model as the team's vector database.

Sepehr's vector database embeds every comment with Yaldat/Yalda-Embedding
(Persian, FP16, 1,024-d output truncated to the first 768 Matryoshka
dimensions, L2-normalised and stored as INT8, as Sepehr confirmed; query prompt for
questions, no prompt for documents).
The comparison only needs the reviews of the two or three products being
compared (at most ~230 each), so instead of loading the full multi-GB index
we embed those reviews on the fly with the same model and the same prompts,
then rank them by cosine similarity to the question. Embeddings are cached
per product, so a second question about the same product is instant.

Model: models/Yalda-Embedding if present, else env YALDA_MODEL_DIR, else
downloaded from Hugging Face (Yaldat/Yalda-Embedding) on first use.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .. import stats
from ..config import ROOT, env

MAX_TOKENS = 512      # same as the team index for comments (products use 128)
MODEL_ID = "Yaldat/Yalda-Embedding"
DIM = 768             # same truncation as the team's vector database
BATCH = 16            # 512 tokens per comment; keeps GPU memory low


@lru_cache(maxsize=1)
def model():
    import torch
    from sentence_transformers import SentenceTransformer

    # Local copy if present; otherwise Hugging Face downloads it once (~2.4 GB).
    local = ROOT / "models" / "Yalda-Embedding"
    path = env("YALDA_MODEL_DIR", str(local) if local.exists() else MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    m = SentenceTransformer(path, device=device, model_kwargs=kwargs)
    m.max_seq_length = MAX_TOKENS
    return m


def _truncate_normalise(vecs: np.ndarray) -> np.ndarray:
    """Keep the first DIM dimensions, then L2-normalise (Sepehr's order)."""
    vecs = vecs[:, :DIM].astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)


def _int8_roundtrip(vecs: np.ndarray) -> np.ndarray:
    """Store-and-reload like the team index: symmetric INT8 with one scale per
    vector (the same scheme as HW03), then back to float for scoring."""
    scales = np.abs(vecs).max(axis=1, keepdims=True).clip(min=1e-12) / 127.0
    q = np.clip(np.round(vecs / scales), -127, 127).astype(np.int8)
    return q.astype(np.float32) * scales


def _doc_text(row) -> str:
    parts = [row.title] if isinstance(row.title, str) and row.title else []
    parts.append(row.body)
    if len(row.advantages):
        parts.append("نقاط قوت: " + "، ".join(row.advantages))
    if len(row.disadvantages):
        parts.append("نقاط ضعف: " + "، ".join(row.disadvantages))
    return " | ".join(parts)


@lru_cache(maxsize=64)
def product_index(product_id: int) -> tuple[np.ndarray, np.ndarray]:
    """(comment ids, unit-normalised document vectors) for one product."""
    df = stats.comments()
    df = df[(df["product_id"] == product_id) & (df["body"].str.len() >= 15)]
    if df.empty:
        return np.array([], dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
    texts = [_doc_text(r) for r in df.itertuples()]
    vecs = model().encode(texts, prompt_name="document", batch_size=BATCH,
                          normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False)
    return df["comment_id"].to_numpy(np.int64), _int8_roundtrip(_truncate_normalise(vecs))


@lru_cache(maxsize=256)
def _query_vector(query: str) -> np.ndarray:
    vec = model().encode([query], prompt_name="query", normalize_embeddings=False,
                         convert_to_numpy=True, show_progress_bar=False)
    return _truncate_normalise(vec)[0]


def retrieve_comment_ids(query: str, product_id: int, k: int) -> list[int]:
    """Adapter with the signature compare.compare() expects."""
    ids, vecs = product_index(int(product_id))
    if not len(ids) or not query.strip():
        return []
    scores = vecs @ _query_vector(query)
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(ids[i]) for i in order]
