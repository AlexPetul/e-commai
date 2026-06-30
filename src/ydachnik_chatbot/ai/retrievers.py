import logging

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ydachnik_chatbot.catalog.csv_reader import read_products_csv
from ydachnik_chatbot.catalog.documents_loader import convert_products_to_bm25_documents
from ydachnik_chatbot.infrastructure.db.vectorstore import get_vectorstore
from ydachnik_chatbot.settings import settings

logger = logging.getLogger(__name__)

VECTOR_SEARCH_TOP_K = 30
BM25_SEARCH_TOP_K = 30

_bm25_retriever: BM25Retriever | None = None


def _load_bm25_documents() -> list[Document]:
    try:
        with open(settings.products_csv_path, encoding="utf-8-sig") as f:
            items = read_products_csv(f.read())
    except FileNotFoundError:
        logger.warning("Products CSV not found at %s, BM25 disabled.", settings.products_csv_path)
        return []
    return convert_products_to_bm25_documents(items)


async def get_bm25_retriever() -> BM25Retriever | None:
    global _bm25_retriever
    if _bm25_retriever is not None:
        return _bm25_retriever

    documents = _load_bm25_documents()
    if not documents:
        logger.warning("No product documents found for BM25 retriever.")
        return None

    _bm25_retriever = BM25Retriever.from_documents(
        documents,
        k=BM25_SEARCH_TOP_K,
        bm25_params={"k1": 1.5, "b": 0.75},
    )
    logger.info("BM25 retriever initialized with %d documents.", len(documents))
    return _bm25_retriever


async def retrieve(
    query: str,
    *,
    hard_filters: dict | None = None,
    category: str | None = None,
) -> list[Document]:
    vectorstore = await get_vectorstore()
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": VECTOR_SEARCH_TOP_K,
            **({"filter": hard_filters} if hard_filters else {}),
        },
    )

    bm25_retriever = await get_bm25_retriever()
    if bm25_retriever is None:
        return await vector_retriever.ainvoke(query)

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],
        id_key="url",
    )
    docs = await ensemble.ainvoke(query)

    if category:
        cat_lower = category.lower()
        docs = [d for d in docs if d.metadata.get("category", "").lower() == cat_lower]

    return docs


async def init_bm25_retriever() -> None:
    await get_bm25_retriever()


async def get_retriever(k: int = 4):
    return (await get_vectorstore()).as_retriever(search_kwargs={"k": k})
