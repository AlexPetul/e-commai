from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_customer_support_context() -> str:
    support_file = Path(__file__).with_name("prompts_data") / "customer_support.md"
    return support_file.read_text(encoding="utf-8").strip()
