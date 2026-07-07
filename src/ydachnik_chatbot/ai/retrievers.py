import logging

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ydachnik_chatbot.catalog.documents_loader import convert_products_to_bm25_documents
from ydachnik_chatbot.infrastructure.db.product_category_repo import product_category_repo
from ydachnik_chatbot.infrastructure.db.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

VECTOR_SEARCH_TOP_K = 30
BM25_SEARCH_TOP_K = 30


async def get_bm25_retriever(category: str) -> BM25Retriever | None:
    items = await product_category_repo.filter_products_by_category(category)
    documents = convert_products_to_bm25_documents(items)

    if not documents:
        logger.warning("No product documents found for BM25 retriever.")
        return None

    bm25_retriever = BM25Retriever.from_documents(
        documents,
        k=BM25_SEARCH_TOP_K,
        bm25_params={"k1": 1.5, "b": 0.75},
    )
    logger.info("BM25 retriever initialized with %d documents.", len(documents))
    return bm25_retriever


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

    bm25_retriever = await get_bm25_retriever(category)
    if bm25_retriever is None:
        return await vector_retriever.ainvoke(query)

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],
        id_key="url",
    )
    docs = await ensemble.ainvoke(query)
    return docs


async def get_retriever(k: int = 4):
    return (await get_vectorstore()).as_retriever(search_kwargs={"k": k})
