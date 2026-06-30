FROM python:3.13-slim AS base

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

ENV PATH="/code/.venv/bin:$PATH"
ENV PYTHONPATH=/code/src


FROM base AS api

COPY src /code/src
COPY .chainlit /code/.chainlit
COPY public /code/public

CMD ["fastapi", "run", "--app", "app", "src/ydachnik_chatbot/app.py"]


FROM base AS migrate

COPY src /code/src
COPY alembic /code/alembic
COPY alembic.ini /code/alembic.ini

ENTRYPOINT ["alembic", "upgrade", "head"]


FROM base AS loaddata

COPY src /code/src

ENTRYPOINT ["python", "-m", "ydachnik_chatbot.management.loaddata"]
CMD ["--filepath", "products.csv"]
