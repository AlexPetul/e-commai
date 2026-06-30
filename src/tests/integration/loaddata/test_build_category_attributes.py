import json

from ydachnik_chatbot.management.loaddata_utils import (
    build_category_attributes,
    build_product_rows,
)
from ydachnik_chatbot.schemas import ProductItem


def test_build_category_attributes():
    category = "Бур"
    item1 = ProductItem(
        url="https://example.com/1",
        title="Item 1",
        category=category,
        attributes=json.dumps(
            [
                {
                    "name": "Длина",
                    "value": 5,
                    "description": "Длина (мм).",
                },
            ]
        ),
    )
    item2 = ProductItem(
        url="https://example.com/2",
        title="Item 2",
        category=category,
        attributes=json.dumps(
            [
                {
                    "name": "Длина",
                    "value": 10,
                    "description": "Длина (мм).",
                },
                {
                    "name": "Глубина",
                    "value": 50,
                    "description": "Глубина (мм).",
                },
            ]
        ),
    )
    result = build_category_attributes([item1, item2])

    assert result == {
        "Бур": [
            {"name": "Длина", "example_values": [5, 10], "description": "Длина (мм)."},
            {
                "name": "Глубина",
                "example_values": [50],
                "description": "Глубина (мм).",
            },
        ]
    }


def test_build_product_rows():
    item = ProductItem(
        url="https://example.com/product",
        title="Product title",
        category="Категория",
        description="Описание",
        tech_specs="Характеристики",
        price="123.45",
        currency="BYN",
        image="https://example.com/image.jpg",
        attributes="[]",
    )

    assert build_product_rows([item]) == [
        {
            "url": "https://example.com/product",
            "title": "Product title",
            "description": "Описание",
            "tech_specs": "Характеристики",
            "price": "123.45",
            "currency": "BYN",
            "category_name": "Категория",
            "image": "https://example.com/image.jpg",
            "attributes": "[]",
        }
    ]
