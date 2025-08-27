FROM python:3.12.3-slim AS base

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

# The app name (APP_NAME) and variant (VARIANT) must be provided as environment variables at runtime (e.g. via docker run -e APP_NAME=Skill_Companion -e VARIANT=default)
ENV APP_NAME=Skill_Companion
ENV VARIANT=default

COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv
# Only install the client dependencies
RUN uv sync --frozen --only-group client

COPY src/client/ ./client/
COPY src/schema/ ./schema/
COPY src/static ./src/static/
COPY src/variants ./variants/
COPY .streamlit/ ./.streamlit/
COPY themes/ ./themes/

# Copy all possible app entrypoints so the runtime env var can select which to run
COPY src/*.py .

CMD sh -c 'uv run theme_selector.py --app "$APP_NAME" --variant "$VARIANT" && uv run streamlit run "$APP_NAME.py"'