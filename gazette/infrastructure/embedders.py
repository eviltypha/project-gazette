"""Offline, deterministic embedder: hashed word + character n-gram TF-IDF.

Character n-grams give the vector retriever fuzzy matching that BM25 lacks
(e.g. "re-KYC'd" vs "periodic updation of KYC", "PEPs" vs "PEP")."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from gazette.infrastructure.text import tokenize

_CHAR_WEIGHT = 0.4


def _bucket(feature: str, dim: int) -> int:
    digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class HashedTfidfEmbedder:
    name = "hashed-tfidf-v1"

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        self._idf = np.ones(dim)

    def _counts(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        tokens = tokenize(text)
        for token in tokens:
            vec[_bucket("w:" + token, self.dim)] += 1.0
            padded = f"<{token}>"
            for n in (3, 4):
                for i in range(len(padded) - n + 1):
                    vec[_bucket("c:" + padded[i : i + n], self.dim)] += _CHAR_WEIGHT
        for left, right in zip(tokens, tokens[1:]):
            vec[_bucket(f"b:{left} {right}", self.dim)] += 1.0
        return vec

    def fit(self, texts: Sequence[str]) -> None:
        counts = np.stack([self._counts(t) for t in texts])
        doc_freq = (counts > 0).sum(axis=0)
        self._idf = np.log((1 + len(texts)) / (1 + doc_freq)) + 1.0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.stack([np.log1p(self._counts(t)) for t in texts]) * self._idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)

    def state(self) -> dict:
        return {"dim": self.dim, "idf": self._idf.tolist()}

    def load_state(self, state: dict) -> None:
        self.dim = state["dim"]
        self._idf = np.asarray(state["idf"])
