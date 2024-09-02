FROM python:3.12.3-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache -r pyproject.toml

COPY src/client/ ./client/
COPY src/schema/ ./schema/
COPY src/streamlit_app.py .

CMD ["streamlit", "run", "streamlit_app.py"]
