from pathlib import Path

import numpy as np
import pytest

from ydachnik_chatbot.ai.category_candidates import CsvCategoryClassifier


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts, *, normalize_embeddings=True):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)

        def vec(text: str):
            t = text.lower()

            if "газонокосил" in t:
                return [1, 0, 0, 0, 0]
            if "триммер" in t:
                return [0, 1, 0, 0, 0]
            if "пила" in t or "бензопил" in t:
                return [0, 0, 1, 0, 0]
            if "мойк" in t:
                return [0, 0, 0, 1, 0]
            if "пылес" in t:
                return [0, 0, 0, 0, 1]

            return [0, 0, 0, 0, 0]

        vectors = np.array([vec(t) for t in items], dtype=np.float32)

        return vectors[0] if single else vectors


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,expected_category",
    [
        ("Нужна газонокосилка", "Газонокосилки"),
        ("хочу триммер", "Триммеры"),
        ("нужна бензопила", "Бензопилы, электропилы"),
        ("мойка высокого давления", "Мойки высокого давления"),
        ("строительный пылесос", "Строительные пылесосы"),
    ],
)
async def test_csv_category_classifier(
    query,
    expected_category,
):
    csv_path = Path(__file__).resolve().parents[3] / "products.csv"
    fake_model = FakeEmbeddingModel()
    classifier = CsvCategoryClassifier(csv_path=csv_path, embedding_model=fake_model)

    candidates = await classifier.get_nearest_category_candidates(query)

    assert expected_category in candidates
