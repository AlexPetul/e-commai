from pydantic import BaseModel, ConfigDict


class ProductItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str
    title: str
    image: str | None = ""
    description: str = ""
    tech_specs: str = ""
    price: float = 0.0
    currency: str = ""
    category: str = ""
    attributes: str = ""


class ProductMetadataSchema(BaseModel):
    url: str
    title: str
    price: float
    currency: str
    category: str = ""
    image: str | None = ""
