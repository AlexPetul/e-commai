from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str = Field(default="postgres")
    postgres_password: SecretStr = Field(default="postgres")  # ty: ignore[invalid-assignment]
    postgres_db: str = Field(default="app_db")
    postgres_host: str = Field(default="db")
    postgres_port: int = Field(default=5432)

    openai_api_key: SecretStr | None = None
    openai_model: str = Field(default="gpt-4o-mini")
    audio_transcription_model: str = Field(default="gpt-4o-mini-transcribe")

    user_id_cookie_name: str = Field(default="ydachnik_user_id")
    user_id_cookie_max_age: int = Field(default=60 * 60 * 24 * 30)

    products_csv_path: str = Field(default="products.csv")

    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)
    vector_table_name: str = Field(default="products_vector")
    category_selection_model: str | None = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    @computed_field
    @property
    def database_url(self) -> str:
        pwd = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg_async://{self.postgres_user}:{pwd}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def postgres_dsn(self) -> str:
        pwd = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg_async://{self.postgres_user}:{pwd}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def checkpointer_dsn(self) -> str:
        pwd = self.postgres_password.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{pwd}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = AppSettings()
