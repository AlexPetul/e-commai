from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ydachnik_chatbot.infrastructure.db.models.base import Base


class ProductCategory(Base):
    __tablename__ = "product_categories"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    attributes: Mapped[list] = mapped_column(JSONB, nullable=False)
    products: Mapped[list["Product"]] = relationship("Product", back_populates="category")


class Product(Base):
    __tablename__ = "product"

    url: Mapped[str] = mapped_column(String(500), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    tech_specs: Mapped[str] = mapped_column(Text, default="", nullable=False)
    price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    currency: Mapped[str] = mapped_column(String(20), default="", nullable=False)
    category_name: Mapped[str] = mapped_column(
        ForeignKey("product_categories.name"),
        nullable=False,
        index=True,
    )
    image: Mapped[str | None] = mapped_column(Text, nullable=True)
    attributes: Mapped[str] = mapped_column(Text, default="", nullable=False)

    category: Mapped["ProductCategory"] = relationship(
        "ProductCategory",
        back_populates="products",
        lazy="joined",
    )
