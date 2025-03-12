FROM python:3.12.3-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

# Copy the project configuration files
# This file defines project metadata and dependencies
COPY pyproject.toml .
# Lock file that ensures consistent dependency versions
COPY uv.lock .

# Install uv, a fast Python package installer and resolver (alternative to pip)
RUN pip install --no-cache-dir uv

# Install only the dependencies needed for the client application
# --frozen: Use exact versions from the lock file
# --only-group client: Only install dependencies marked as part of the "client" group in pyproject.toml
# This keeps the Docker image smaller by excluding unnecessary dependencies
RUN uv sync --frozen --only-group client

COPY src/client/ ./client/
COPY src/schema/ ./schema/
COPY src/streamlit_app.py .

CMD ["streamlit", "run", "streamlit_app.py"]
