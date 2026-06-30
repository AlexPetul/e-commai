import json
import math
from collections import defaultdict

from ydachnik_chatbot.schemas import ProductItem


def _example_value_sort_key(value):
    if isinstance(value, bool):
        return (2, str(value))
    if isinstance(value, (int, float)):
        return (0, float(value), str(value))
    return (1, str(value))


def build_category_attributes(items: list[ProductItem]) -> dict:
    result = defaultdict(dict)

    for item in items:
        if not item.category or not item.attributes or item.attributes == "[]":
            continue

        try:
            parsed_attrs = json.loads(item.attributes)
        except (json.JSONDecodeError, ValueError):
            continue

        category_attrs = result[item.category]

        for attr in parsed_attrs:
            name = attr["name"]
            value = attr["value"]

            if isinstance(value, float) and math.isnan(value):
                continue

            if name not in category_attrs:
                category_attrs[name] = {
                    "name": name,
                    "example_values": set(),
                    "description": attr["description"],
                }

            category_attrs[name]["example_values"].add(value)

    return {
        category: [
            {
                **attr,
                "example_values": sorted(attr["example_values"], key=_example_value_sort_key)[:5],
            }
            for attr in attrs.values()
        ]
        for category, attrs in result.items()
    }


def build_product_rows(items: list[ProductItem]) -> list[dict]:
    rows: list[dict] = []
    for item in items:
        if not item.category:
            continue

        rows.append(
            {
                "url": item.url,
                "title": item.title,
                "description": item.description,
                "tech_specs": item.tech_specs,
                "price": item.price,
                "currency": item.currency,
                "category_name": item.category,
                "image": item.image or None,
                "attributes": item.attributes,
            }
        )

    return rows
