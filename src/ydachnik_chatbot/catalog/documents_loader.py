import json
import math
from collections.abc import Sequence

from langchain_core.documents import Document

from ydachnik_chatbot.schemas import ProductItem, ProductMetadataSchema


def _parse_attributes(raw: str) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return {
                attr["name"]: attr["value"]
                for attr in parsed
                if isinstance(attr, dict) and "name" in attr and "value" in attr
            }
        return {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _clean(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def build_metadata(item: ProductItem) -> dict:
    metadata = ProductMetadataSchema(
        url=item.url,
        title=item.title,
        price=float(item.price or 0),
        currency=item.currency,
        category=item.category,
        image=item.image,
    ).model_dump()
    metadata.update({k: _clean(v) for k, v in _parse_attributes(item.attributes).items()})
    return metadata


def convert_products_to_documents(items: Sequence[ProductItem]) -> list[Document]:
    return [
        Document(
            id=item.url,
            page_content=item.description,
            metadata=build_metadata(item),
        )
        for item in items
        if item.category
    ]


def convert_products_to_bm25_documents(items: Sequence[ProductItem]) -> list[Document]:
    docs = []
    for item in items:
        content = " ".join(filter(None, [item.title, item.description, item.tech_specs]))
        if not content:
            continue
        docs.append(Document(page_content=content, metadata=build_metadata(item)))
    return docs
