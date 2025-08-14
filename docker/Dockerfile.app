FROM python:3.12.3-slim AS base

ARG STREAMLIT_PAGE_NAME="Agent_Service_Toolkit.py"
# this shit format is necessary, since ARG vars are used in build-time and ENV vars in runtime. Noice.
ENV STREAMLIT_START_PAGE_NAME=${STREAMLIT_PAGE_NAME}
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
COPY src/static ./src/static/
COPY .streamlit/ ./.streamlit/

# INFO: Build separated into 2 stages. => dev stage takes the Agent_Service_Toolkit.py file in root and sees all other pages as sub-pages for easier development.
FROM base AS dev
COPY src/pages/ ./pages/
COPY src/${STREAMLIT_PAGE_NAME} .
CMD streamlit run ${STREAMLIT_START_PAGE_NAME}

# prod stage takes only one specified page from the pages/ directory and sets it into root. This enables the deployment of single pages as own application.
FROM base AS prod
COPY src/pages/${STREAMLIT_PAGE_NAME} .
CMD streamlit run ${STREAMLIT_START_PAGE_NAME}