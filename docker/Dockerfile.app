FROM python:3.12.3-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv
# Only install the client dependencies
RUN uv sync --frozen --only-group client

COPY src/client/ ./client/
COPY src/schema/ ./schema/
COPY src/streamlit_app.py .

CMD ["streamlit", "run", "streamlit_app.py"]
