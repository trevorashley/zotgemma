"""EmbeddingGemma 2 wrapper: lazy load, MPS, bf16/fp32 selection, prompt handling."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from . import config

log = logging.getLogger(__name__)

QUERY_PROMPT_NAME = "SearchQuery"
_SIMILAR = ("How do I stabilize a nonlinear system with a Lyapunov function?",
            "A Lyapunov function can be used to prove stability of a nonlinear control system.")
_UNRELATED = "The recipe calls for two cups of flour and a pinch of salt."


def build_document(title: str, body: str) -> str:
    """Build the document string ``title: {title} | text: {body}`` by hand.

    The built-in ``Document`` prompt hardcodes ``title: none``, so documents are
    encoded without a prompt name and carry their real title instead.
    """
    return f"title: {title.strip() or 'none'} | text: {body.strip()}"


def normalize_rows(a: np.ndarray) -> np.ndarray:
    """L2-normalize each row (used after Matryoshka truncation)."""
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-12)


def truncate(a: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka-truncate to ``dim`` and re-normalize."""
    return normalize_rows(a[..., :dim]) if dim < a.shape[-1] else a


@dataclass(slots=True)
class EmbedderInfo:
    """What the embedder actually ended up using."""

    device: str
    dtype: str
    sanity: str


def _load_model(model_id: str, device: str, dtype):
    """Load the text-only model, preferring the local HF cache (no Hub traffic) and
    falling back to a download if it is not cached yet."""
    from sentence_transformers import SentenceTransformer
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    kwargs = dict(device=device, model_kwargs={"dtype": dtype},
                  config_kwargs={"vision_config": None, "audio_config": None})  # text-only (270M)
    try:
        return SentenceTransformer(model_id, local_files_only=True, **kwargs)
    except (OSError, ValueError):
        log.info("model not in local cache; downloading %s", model_id)
        return SentenceTransformer(model_id, **kwargs)


class Embedder:
    """Loads the model once and embeds queries and documents.

    Tries bf16 on the accelerator first; falls back to fp32 if outputs are
    non-finite or the similarity sanity check fails. fp16 is never used.
    """

    def __init__(self, model_id: str = config.MODEL_ID) -> None:
        import torch
        self.model_id = model_id
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        candidates = [torch.bfloat16, torch.float32] if device != "cpu" else [torch.float32]
        self.info: EmbedderInfo | None = None
        last_err = "no candidate dtype tried"
        for dtype in candidates:
            model = _load_model(model_id, device, dtype)
            model.max_seq_length = config.MAX_SEQ_LENGTH
            self._model = model
            ok, msg = self._sanity_check()
            if ok:
                self.info = EmbedderInfo(device, str(dtype).removeprefix("torch."), msg)
                break
            last_err = f"{dtype}: {msg}"
            log.warning("Embedder sanity check failed (%s); trying next dtype", last_err)
        if self.info is None:
            raise RuntimeError(f"Embedding model {model_id} failed its sanity check: {last_err}")

    def _sanity_check(self) -> tuple[bool, str]:
        q = self.embed_query(_SIMILAR[0])
        d = self.embed_documents_raw([_SIMILAR[1], _UNRELATED])
        if not (np.isfinite(q).all() and np.isfinite(d).all()):
            return False, "non-finite embeddings"
        s_pos, s_neg = float(q @ d[0]), float(q @ d[1])
        if not s_pos > s_neg:
            return False, f"similar pair {s_pos:.3f} <= unrelated {s_neg:.3f}"
        return True, f"similar={s_pos:.3f} unrelated={s_neg:.3f}"

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query (``SearchQuery`` prompt), unit-normalized float32, 768-d."""
        v = self._model.encode(text, prompt_name=QUERY_PROMPT_NAME, convert_to_numpy=True,
                               normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def embed_documents_raw(self, texts: list[str], batch_size: int = config.EMBED_BATCH_SIZE) -> np.ndarray:
        """Embed pre-built document strings with no prompt (see :func:`build_document`)."""
        v = self._model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def count_tokens(self, texts: list[str]) -> int:
        """Total token count (after truncation to max_seq_length) for throughput reporting."""
        tok = self._model.tokenizer
        enc = tok(texts, truncation=True, max_length=config.MAX_SEQ_LENGTH, add_special_tokens=True)
        return int(sum(len(x) for x in enc["input_ids"]))


_embedder: Embedder | None = None
_lock = threading.Lock()


def get_embedder() -> Embedder:
    """Return the process-wide embedder, constructing it on first use."""
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = Embedder()
        return _embedder
