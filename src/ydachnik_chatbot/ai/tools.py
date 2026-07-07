import logging
from datetime import timedelta
from typing import Annotated, Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from cachier import cachier
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from ydachnik_chatbot.ai.category_candidates import get_nearest_category_candidates
from ydachnik_chatbot.ai.filters import score_document
from ydachnik_chatbot.ai.retrievers import retrieve
from ydachnik_chatbot.infrastructure.db.product_category_repo import product_category_repo

logger = logging.getLogger(__name__)


@tool
async def load_category_attributes(
    category: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Load the list of searchable attributes for the given product category.

    Call this once when the product category is known to better understand
    possible attributes of the product to search for.
    """
    attributes = await product_category_repo.fetch_category_attributes(category)
    attributes.append(
        {
            "name": "price",
            "description": "Стоимость в BYN",
        }
    )

    content = (
        f"Attributes for category '{category}': {attributes}"
        if attributes
        else f"Attributes for category '{category}' not found."
    )

    return Command(
        update={
            "product_attribute_schema": list(attributes),
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )


@tool
async def save_product_attributes(
    attributes: dict[str, Any],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Save the extracted attributes of the product in state.

    Call this once you receive from the user any details about
    the product he is searching for.
    """
    return Command(
        update={
            "product_attributes": {
                key: str(value) for key, value in attributes.items() if value is not None
            },
            "messages": [
                ToolMessage(
                    content=f"Product attributes extracted: {attributes}", tool_call_id=tool_call_id
                )
            ],
        }
    )


@tool
async def set_product_category(
    runtime: ToolRuntime,
    category: str | None,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Set the current product category in state.

    Call this after deciding which category best matches the user's request.
    """
    current_category = runtime.state.get("product_category")
    content = f"Product category set to: {category}" if category else "Product category cleared."
    update: dict[str, Any] = {"product_category": category}

    if category != current_category:
        update["product_attribute_schema"] = None
        # update["product_attributes"] = None

    return Command(
        update={**update, "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}
    )


@tool
async def get_category_candidates_for_classification(runtime: ToolRuntime) -> list[str]:
    """Get list of categories that match user's query the best.

    Call this only when the product category is unknown or unclear.
    """
    message = [
        m.content.strip()
        for m in runtime.state["messages"]
        if isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.strip()
    ][-1]

    return await get_nearest_category_candidates(message)


@tool
async def get_price_range(category: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Get price range for product category and product preferences."""
    min_p, max_p = await product_category_repo.get_price_range(category=category)

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Price range is {min_p} to {max_p} BYN", tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
async def search_products(
    query: str,
    category: str,
    filters: dict,
) -> str:
    """Search the product catalog for items matching the query.

    :param query: do not pass the user's raw message. Instead, rewrite it into
      natural-language product description combining category and product preferences (attributes).

    :param category: product category name

    :param filters: metadata filters using operators, e.g.:
      {"engine_type": {"$ilike": "бензиновая"}, "mulching_capability": {"$eq": true},
       "max_cutting_height_mm": {"$gte": 60}}
    Supported operators: $eq, $ne, $lt, $lte, $gt, $gte, $in, $nin, $like, $ilike.
    Only include filters for attributes the user explicitly specified.
    """
    conditions: list[dict] = [
        {"category": {"$ilike": category}},
        {"price": {"$ne": 0.0}},
    ]
    if price := filters.pop("price", None):
        conditions.append({"price": {op: float(v) for op, v in price.items()}})

    hard_filters = {"$and": conditions}

    docs = await retrieve(query, hard_filters=hard_filters, category=category)

    scored = sorted(
        ((score_document(doc, filters or {}), doc) for doc in docs),
        key=lambda x: x[0],
        reverse=True,
    )

    result = []
    for score, doc in scored[:5]:
        result.append(
            "\n".join(
                (
                    f"Score: {score:.3f}",
                    f"Content: {doc.page_content}",
                    f"Data: {doc.metadata}",
                )
            )
        )
    return "\n------\n".join(result)


@tool
@cachier(backend="memory", stale_after=timedelta(hours=1))
async def fetch_page(url: str) -> str:
    """Fetch the <main> block of a website page and return readable text."""
    parsed = urlparse(url)
    if parsed.netloc not in {"ydachnik.by", "www.ydachnik.by"}:
        return "Unsupported domain. Use only ydachnik.by pages."

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.exception("Failed to fetch page: %s", url)
        return f"Failed to fetch page: {exc}"

    soup = BeautifulSoup(response.text, "lxml")
    main = soup.select_one("main")
    if main is None:
        return "No <main> block found on the page."

    title = soup.title.get_text(strip=True) if soup.title else ""
    text = main.get_text(separator="\n", strip=True)
    return "\n".join(part for part in [f"Title: {title}" if title else "", text] if part)
