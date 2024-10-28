FROM python:3.12.3-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv
# RUN uv sync --frozen --no-install-project --no-dev
# Test expected deps in Streamlit Cloud
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY src/client/ ./client/
COPY src/schema/ ./schema/
COPY src/streamlit_app.py .

CMD ["streamlit", "run", "streamlit_app.py"]
