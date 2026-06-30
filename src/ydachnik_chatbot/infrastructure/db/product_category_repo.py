from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ydachnik_chatbot.infrastructure.db.models.product import Product, ProductCategory
from ydachnik_chatbot.schemas import ProductItem
from ydachnik_chatbot.settings import settings

_engine: AsyncEngine | None = None


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(settings.database_url)
    return _engine


async def fetch_category_attributes(category: str) -> list[dict]:
    async with _get_engine().connect() as conn:
        result = await conn.execute(
            select(ProductCategory.attributes).where(ProductCategory.name == category)
        )
        row = result.fetchone()
        return list(row[0]) if row else []


async def fetch_category_names() -> list[str]:
    async with _get_engine().connect() as conn:
        result = await conn.execute(select(ProductCategory.name).order_by(ProductCategory.name))
        return list(result.scalars().all())


async def fetch_products_for_category_selection() -> list[ProductItem]:
    async with _get_engine().connect() as conn:
        result = await conn.execute(
            select(
                Product.url,
                Product.title,
                Product.category_name,
            )
            .where(Product.category_name != "")
            .order_by(Product.category_name, Product.title)
        )

        return [
            ProductItem(
                url=row.url,
                title=row.title,
                category=row.category_name,
            )
            for row in result.fetchall()
        ]
