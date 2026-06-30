__all__ = ["init_vectorstore_table", "init_vector_store", "get_vectorstore", "get_engine"]


def get_engine(*args, **kwargs):
    from ydachnik_chatbot.infrastructure.db.engine import get_engine as _get_engine

    return _get_engine(*args, **kwargs)


def init_vectorstore_table(*args, **kwargs):
    from ydachnik_chatbot.infrastructure.db.engine import (
        init_vectorstore_table as _init_vectorstore_table,
    )

    return _init_vectorstore_table(*args, **kwargs)


def get_vectorstore(*args, **kwargs):
    from ydachnik_chatbot.infrastructure.db.vectorstore import get_vectorstore as _get_vectorstore

    return _get_vectorstore(*args, **kwargs)


def init_vector_store(*args, **kwargs):
    from ydachnik_chatbot.infrastructure.db.vectorstore import (
        init_vector_store as _init_vector_store,
    )

    return _init_vector_store(*args, **kwargs)
