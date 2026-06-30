# ydachnik-chatbot

`ydachnik-chatbot` is an AI assistant for ydachnik.by that helps users find and compare products from the catalog. It combines a chat UI, product search, and LLM-driven conversation flow, with support for text and voice input.

## Tech Stack

- `Python 3.13`
- `FastAPI`
- `Chainlit`
- `LangGraph`
- `LangChain`
- `PostgreSQL`
- `pgvector` / vector search via LangChain Postgres integration
- `OpenAI API` for chat and embeddings
- `BM25` lexical retrieval
- `BeautifulSoup`, `lxml`, `Scrapy`, `httpx` for catalog and page processing
- `pytest`, `ruff`, `ty` for testing and code quality
- `deepeval` for LLM evaluation tests

## What it does

- Helps users choose products from the ydachnik.by catalog
- Routes conversations between product consultation and support
- Uses catalog data, vector search, and keyword retrieval to answer questions
- Supports speech-to-text input for voice messages

## Tests

- Test suite is based on `pytest`
- Default `pytest` runs exclude AI/evaluation tests because the repository is configured with `-m 'not ai'`
- Unit and integration tests cover catalog parsing, category selection, and data-loading logic
- AI-marked tests validate conversation routing and LLM-driven flows when the required configuration is available
