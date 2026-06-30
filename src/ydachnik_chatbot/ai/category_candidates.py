from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
from rapidfuzz import fuzz

from ydachnik_chatbot.catalog.csv_reader import read_products_csv
from ydachnik_chatbot.schemas import ProductItem
from ydachnik_chatbot.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    def encode(
        self,
        texts: str | Sequence[str],
        *,
        normalize_embeddings: bool = True,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class CategoryStats:
    embeddings: np.ndarray
    centroid: np.ndarray
    source_texts: tuple[str, ...]


@dataclass(frozen=True)
class CategoryIndex:
    stats: dict[str, CategoryStats]


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _build_query_text(query: str, current_category: str | None) -> str:
    parts = [_normalize_text(query)]
    if current_category:
        parts.append(_normalize_text(current_category))
    return " ".join(part for part in parts if part).strip()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _category_text_variants(category: str) -> list[str]:
    normalized = _normalize_text(category)
    lower = normalized.lower()
    compact = re.sub(r"\s+", " ", lower)
    variants = [
        normalized,
        lower,
        f"категория {compact}",
        f"товар категории {compact}",
    ]
    return _dedupe_preserve_order([variant for variant in variants if variant])


def build_category_corpus(
    items: Sequence[ProductItem],
    *,
    per_category: int = 10,
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for item in items:
        category = _normalize_text(item.category)
        title = _normalize_text(item.title)
        if not category or not title:
            continue

        grouped.setdefault(category, [])
        if len(grouped[category]) < per_category:
            grouped[category].append(title)

    return {category: tuple(titles) for category, titles in grouped.items()}


def _get_embedding_model():
    model_name = settings.category_selection_model
    if not model_name:
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        logger.warning(
            "sentence_transformers is not installed; falling back to fuzzy category matching."
        )
        return None

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        logger.warning(
            "Failed to initialize category embedding model %s; falling back to fuzzy matching: %s",
            model_name,
            exc,
        )
        return None


def _build_category_stats(
    corpus: dict[str, tuple[str, ...]],
    embedding_model: EmbeddingModel | None,
) -> dict[str, CategoryStats]:
    if embedding_model is None or not corpus:
        return {}

    stats: dict[str, CategoryStats] = {}
    for category, titles in corpus.items():
        texts = _dedupe_preserve_order(
            [
                *_category_text_variants(category),
                *(f"{category}: {title}" for title in titles[:10]),
            ]
        )
        embeddings = np.asarray(embedding_model.encode(texts, normalize_embeddings=True))
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        centroid = _normalize_embedding(embeddings.mean(axis=0))
        stats[category] = CategoryStats(
            embeddings=embeddings,
            centroid=centroid,
            source_texts=tuple(texts),
        )

    return stats


def build_category_index(
    items: Sequence[ProductItem],
    *,
    embedding_model: EmbeddingModel | None = None,
    per_category: int = 10,
) -> CategoryIndex:
    corpus = build_category_corpus(items, per_category=per_category)
    model = embedding_model if embedding_model is not None else _get_embedding_model()
    return CategoryIndex(stats=_build_category_stats(corpus, model))


def _score_with_embeddings(
    query: str,
    category_stats: dict[str, CategoryStats],
    embedding_model: EmbeddingModel | None,
) -> list[str]:
    if embedding_model is None or not category_stats:
        return []

    query_embedding = np.asarray(embedding_model.encode(query, normalize_embeddings=True))
    scored: list[tuple[str, float]] = []
    for category, stats in category_stats.items():
        centroid_score = float(np.dot(query_embedding, stats.centroid))
        nearest_score = float(np.max(stats.embeddings @ query_embedding))
        combined_score = 0.4 * centroid_score + 0.6 * nearest_score
        scored.append((category, combined_score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [category for category, _ in scored]


def _score_with_fuzz(query: str, categories: Sequence[str]) -> list[str]:
    scored = [
        (
            category,
            max(
                fuzz.token_set_ratio(query, category),
                fuzz.partial_ratio(query, category),
                fuzz.ratio(query, category),
            ),
        )
        for category in categories
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [category for category, _ in scored]


def _load_products_from_csv(csv_path: str | Path) -> list[ProductItem]:
    path = Path(csv_path)
    content = path.read_text(encoding="utf-8-sig")
    return read_products_csv(content)


class BaseCategoryClassifier(ABC):
    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel | None = None,
        per_category: int = 10,
    ) -> None:
        self._embedding_model = (
            embedding_model if embedding_model is not None else _get_embedding_model()
        )
        self._per_category = per_category
        self._index: CategoryIndex | None = None

    @abstractmethod
    async def load_products(self) -> Sequence[ProductItem]:
        raise NotImplementedError

    async def build_index(self) -> CategoryIndex:
        items = await self.load_products()
        self._index = build_category_index(
            items,
            embedding_model=self._embedding_model,
            per_category=self._per_category,
        )
        return self._index

    async def _ensure_index(self) -> CategoryIndex:
        if self._index is None:
            return await self.build_index()
        return self._index

    async def get_nearest_category_candidates(
        self,
        query: str,
        *,
        current_category: str | None = None,
        limit: int = 5,
    ) -> list[str]:
        index = await self._ensure_index()
        if not index.stats:
            return []

        candidate_query = _build_query_text(query, current_category)

        def _resolve() -> list[str]:
            selected = _score_with_embeddings(
                candidate_query,
                index.stats,
                self._embedding_model,
            )
            if not selected:
                selected = _score_with_fuzz(candidate_query, list(index.stats.keys()))

            if current_category:
                selected = [current_category, *selected]

            return _dedupe_preserve_order(selected)[:limit]

        return await asyncio.to_thread(_resolve)


class CsvCategoryClassifier(BaseCategoryClassifier):
    def __init__(
        self,
        csv_path: str | Path,
        *,
        embedding_model: EmbeddingModel | None = None,
        per_category: int = 10,
    ) -> None:
        super().__init__(embedding_model=embedding_model, per_category=per_category)
        self._csv_path = Path(csv_path)

    async def load_products(self) -> Sequence[ProductItem]:
        if not self._csv_path.exists():
            logger.warning("Category CSV not found: %s", self._csv_path)
            return []
        return await asyncio.to_thread(_load_products_from_csv, self._csv_path)


class DbCategoryClassifier(BaseCategoryClassifier):
    async def load_products(self) -> Sequence[ProductItem]:
        from ydachnik_chatbot.infrastructure.db.product_category_repo import (
            fetch_products_for_category_selection,
        )

        return await fetch_products_for_category_selection()


@lru_cache(maxsize=4)
def _get_default_csv_classifier(csv_path: str, per_category: int) -> CsvCategoryClassifier:
    return CsvCategoryClassifier(csv_path=csv_path, per_category=per_category)


async def get_nearest_category_candidates(
    query: str,
    *,
    current_category: str | None = None,
    limit: int = 5,
    csv_path: str | Path | None = None,
    classifier: BaseCategoryClassifier | None = None,
    per_category: int = 10,
) -> list[str]:
    if classifier is None:
        classifier = _get_default_csv_classifier(
            str(Path(csv_path or settings.products_csv_path)),
            per_category,
        )

    return await classifier.get_nearest_category_candidates(
        query,
        current_category=current_category,
        limit=limit,
    )


async def warmup_category_selector(*, csv_path: str | Path | None = None) -> None:
    path = Path(csv_path or settings.products_csv_path)
    if not path.exists():
        logger.warning("No products CSV found during category selector warmup: %s", path)
        return

    classifier = _get_default_csv_classifier(str(path), 10)
    items = await classifier.load_products()
    classifier._index = build_category_index(
        items,
        embedding_model=classifier._embedding_model,
        per_category=classifier._per_category,
    )
    logger.info(
        "Category selector warmed up with %d products across %d categories.",
        len(items),
        len(classifier._index.stats if classifier._index else {}),
    )
