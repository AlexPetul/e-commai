import asyncio
import logging
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

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
    return [category]


def build_category_corpus(
    items: Sequence[ProductItem],
    *,
    per_category: int = 10,
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, None] = {}
    for item in items:
        category = _normalize_text(item.category)
        if not category:
            continue

        grouped.setdefault(category, None)

    return {category: (category,) for category in grouped}


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
    for category in corpus:
        texts = _dedupe_preserve_order(_category_text_variants(category))
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
        limit: int = 5,
    ) -> list[str]:
        index = await self._ensure_index()
        if not index.stats:
            return []

        def _resolve() -> list[str]:
            selected = _score_with_embeddings(
                query,
                index.stats,
                self._embedding_model,
            )

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
            product_category_repo,
        )

        return await product_category_repo.fetch_products_for_category_selection()


@lru_cache(maxsize=4)
def _get_default_csv_classifier(csv_path: str, per_category: int) -> CsvCategoryClassifier:
    return CsvCategoryClassifier(csv_path=csv_path, per_category=per_category)


async def get_nearest_category_candidates(
    query: str,
    *,
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
        limit=limit,
    )


async def warmup_category_selector() -> None:
    path = Path(settings.products_csv_path)
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
