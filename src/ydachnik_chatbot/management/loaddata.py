import argparse
import asyncio
import json
import logging

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import create_async_engine

from ydachnik_chatbot.catalog.csv_reader import read_products_csv
from ydachnik_chatbot.catalog.documents_loader import build_metadata, convert_products_to_documents
from ydachnik_chatbot.infrastructure.db.engine import get_engine, init_vectorstore_table
from ydachnik_chatbot.infrastructure.db.models.product import Product, ProductCategory
from ydachnik_chatbot.infrastructure.db.vectorstore import get_vectorstore, init_vector_store
from ydachnik_chatbot.management.loaddata_utils import build_category_attributes, build_product_rows
from ydachnik_chatbot.schemas import ProductItem
from ydachnik_chatbot.settings import AppSettings


async def seed_product_categories(settings: AppSettings, items: list[ProductItem]) -> None:
    logger = logging.getLogger(__name__)
    seed = build_category_attributes(items)
    async_engine = create_async_engine(settings.database_url)
    try:
        async with async_engine.begin() as conn:
            for category, attributes in seed.items():
                stmt = (
                    insert(ProductCategory)
                    .values(name=category, attributes=attributes)
                    .on_conflict_do_update(
                        index_elements=["name"],
                        set_={"attributes": attributes},
                    )
                )
                await conn.execute(stmt)
        logger.info("Seeded %d product_categories rows.", len(seed))
    finally:
        await async_engine.dispose()


async def seed_products(settings: AppSettings, items: list[ProductItem]) -> None:
    logger = logging.getLogger(__name__)
    rows = build_product_rows(items)
    async_engine = create_async_engine(settings.database_url)
    try:
        async with async_engine.begin() as conn:
            if rows:
                stmt = insert(Product).values(rows)
                update_columns = {
                    column.name: getattr(stmt.excluded, column.name)
                    for column in Product.__table__.columns
                    if not column.primary_key
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=["url"],
                    set_=update_columns,
                )
                await conn.execute(stmt)
        logger.info("Seeded %d product rows.", len(rows))
    finally:
        await async_engine.dispose()


async def recreate_metadata(settings: AppSettings, items: list[ProductItem]) -> None:
    logger = logging.getLogger(__name__)
    async_engine = create_async_engine(settings.database_url)
    table = settings.vector_table_name
    updated = 0
    try:
        async with async_engine.begin() as conn:
            for item in items:
                metadata = build_metadata(item)
                result = await conn.execute(
                    text(
                        f"UPDATE {table} SET langchain_metadata = CAST(:metadata AS jsonb)"  # noqa: S608
                        " WHERE langchain_id = :url"
                    ),
                    {"metadata": json.dumps(metadata), "url": item.url},
                )
                updated += result.rowcount
    finally:
        await async_engine.dispose()

    logger.info("Updated metadata for %d document rows.", updated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load products CSV into pgvector.")
    parser.add_argument("--filepath", default="products.csv", help="Path to products CSV file.")
    parser.add_argument(
        "--recreate-metadata",
        action="store_true",
        help="Only update metadata on existing documents; skip re-embedding.",
    )
    return parser.parse_args()


async def amain() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    args = parse_args()
    settings = AppSettings()

    if args.recreate_metadata:
        with open(args.filepath, encoding="utf-8-sig") as f:
            items = read_products_csv(f.read())
        logger.info("Read %d products.", len(items))
        await recreate_metadata(settings, items)
        logger.info("Seeding product_categories...")
        await seed_product_categories(settings, items)
        logger.info("Seeding products...")
        await seed_products(settings, items)
        logger.info("Done.")
        return

    engine = get_engine(settings)
    try:
        logger.info("Initializing vector table '%s'...", settings.vector_table_name)
        await init_vectorstore_table(
            engine,
            settings.vector_table_name,
            settings.embedding_dimensions,
            overwrite_existing=True,
        )
        await init_vector_store(engine, overwrite_existing=True)

        logger.info("Reading %s...", args.filepath)
        with open(args.filepath, encoding="utf-8-sig") as f:
            items = read_products_csv(f.read())
        logger.info("Read %d products.", len(items))

        documents = convert_products_to_documents(items)
        logger.info("Persisting %d document chunks...", len(documents))

        vectorstore = await get_vectorstore()
        await vectorstore.aadd_documents(documents)

        logger.info("Seeding product_categories...")
        await seed_product_categories(settings, items)

        logger.info("Seeding products...")
        await seed_products(settings, items)

        logger.info("Done.")
    finally:
        await engine.close()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
