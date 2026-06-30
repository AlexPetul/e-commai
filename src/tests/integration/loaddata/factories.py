from polyfactory.factories.pydantic_factory import ModelFactory

from ydachnik_chatbot.schemas import ProductItem


class ProductItemFactory(ModelFactory[ProductItem]):
    pass
