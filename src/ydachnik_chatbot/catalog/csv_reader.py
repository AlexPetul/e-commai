import csv
import io

from ydachnik_chatbot.schemas import ProductItem


def read_products_csv(content: str) -> list[ProductItem]:
    items: list[ProductItem] = []
    for row in csv.DictReader(io.StringIO(content), delimiter=";"):
        item = ProductItem.model_validate(row)
        if item.title:
            items.append(item)
    return items
