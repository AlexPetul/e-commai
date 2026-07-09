from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ydachnik_chatbot.infrastructure.db.models.product import Product, ProductCategory
from ydachnik_chatbot.schemas import ProductItem
from ydachnik_chatbot.settings import settings


class ProductCategoryRepository:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine

    def _get_engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(settings.database_url)
        return self._engine

    async def fetch_category_attributes(self, category: str) -> list[dict]:
        async with self._get_engine().connect() as conn:
            result = await conn.execute(
                select(ProductCategory.attributes).where(ProductCategory.name == category)
            )
            row = result.fetchone()
            return list(row[0]) if row else []

    async def fetch_category_names(self) -> list[str]:
        async with self._get_engine().connect() as conn:
            result = await conn.execute(select(ProductCategory.name).order_by(ProductCategory.name))
            return list(result.scalars().all())

    async def fetch_products_for_category_selection(self) -> list[ProductItem]:
        async with self._get_engine().connect() as conn:
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

    async def filter_products_by_category(self, category: str) -> list[ProductItem]:
        async with self._get_engine().connect() as conn:
            result = await conn.execute(
                select(
                    Product.url,
                    Product.title,
                    Product.category_name,
                    Product.image,
                    Product.description,
                    Product.price,
                    Product.tech_specs,
                    Product.attributes,
                ).where(Product.category_name == category)
            )
            rows = result.fetchall()

        return [
            ProductItem(
                url=row.url,
                title=row.title,
                category=row.category_name,
                image=row.image,
                description=row.description,
                price=row.price,
                tech_specs=row.tech_specs,
                attributes=row.attributes,
            )
            for row in rows
        ]

    async def get_price_range(self, category: str) -> tuple[float, float]:
        async with self._get_engine().connect() as conn:
            result = await conn.execute(
                select(
                    func.min(Product.price),
                    func.max(Product.price),
                )
                .where(Product.category_name == category)
                .group_by(Product.category)
            )

            min_price, max_price = result.one()
            return min_price, max_price


product_category_repo = ProductCategoryRepository()
