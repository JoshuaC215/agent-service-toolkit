from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import docx2txt
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from core import settings

SUPPORTED_EXTENSIONS = {".docx", ".md", ".pdf", ".txt"}
KNOWLEDGE_BASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,55}$")


class KnowledgeBaseError(ValueError):
    """Raised when a knowledge-base operation cannot be completed safely."""


@dataclass(frozen=True)
class IngestResult:
    knowledge_base_id: str
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    source: str
    page: int | None
    content: str
    score: float
    metadata: dict[str, Any]


def _validate_knowledge_base_id(knowledge_base_id: str) -> str:
    if not KNOWLEDGE_BASE_ID_PATTERN.fullmatch(knowledge_base_id):
        raise KnowledgeBaseError(
            "knowledge_base_id must contain 1-56 letters, numbers, hyphens, or underscores "
            "and must start with a letter or number."
        )
    return knowledge_base_id


def _validate_filename(filename: str) -> tuple[str, str]:
    safe_filename = Path(filename).name
    extension = Path(safe_filename).suffix.lower()
    if not safe_filename or extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise KnowledgeBaseError(f"Unsupported document type. Supported types: {supported}.")
    return safe_filename, extension


def _extract_documents(filename: str, content: bytes) -> list[Document]:
    safe_filename, extension = _validate_filename(filename)
    if not content:
        raise KnowledgeBaseError("The uploaded document is empty.")

    if extension in {".md", ".txt"}:
        text = content.decode("utf-8-sig", errors="replace").strip()
        documents = [Document(page_content=text, metadata={"page": 1})] if text else []
    elif extension == ".pdf":
        reader = PdfReader(BytesIO(content))
        documents = [
            Document(page_content=page_text, metadata={"page": page_number})
            for page_number, page in enumerate(reader.pages, start=1)
            if (page_text := (page.extract_text() or "").strip())
        ]
    else:
        temporary_path: str | None = None
        try:
            with NamedTemporaryFile(suffix=".docx", delete=False) as temporary_file:
                temporary_file.write(content)
                temporary_path = temporary_file.name
            text = (docx2txt.process(temporary_path) or "").strip()
            documents = [Document(page_content=text, metadata={"page": 1})] if text else []
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

    if not documents:
        raise KnowledgeBaseError(f"No readable text was found in {safe_filename}.")
    return documents


def _embedding_function() -> Embeddings:
    if settings.USE_FAKE_MODEL:
        # This keeps local demos and tests offline. Use a real embedding provider in production.
        return DeterministicFakeEmbedding(size=settings.KNOWLEDGE_EMBEDDING_DIMENSIONS)
    if not settings.OPENAI_API_KEY:
        raise KnowledgeBaseError(
            "OPENAI_API_KEY is required for document embeddings. "
            "Set USE_FAKE_MODEL=true for an offline demo only."
        )
    return OpenAIEmbeddings(model=settings.KNOWLEDGE_EMBEDDING_MODEL)


def _vector_store(knowledge_base_id: str) -> Chroma:
    safe_id = _validate_knowledge_base_id(knowledge_base_id)
    base_dir = Path(settings.KNOWLEDGE_BASE_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = base_dir / safe_id
    storage_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=f"kb_{safe_id}",
        persist_directory=str(storage_dir),
        embedding_function=_embedding_function(),
    )


def ingest_document(
    knowledge_base_id: str,
    filename: str,
    content: bytes,
    content_type: str | None = None,
) -> IngestResult:
    """Parse, split, and persist one document in a Chroma knowledge base."""
    safe_id = _validate_knowledge_base_id(knowledge_base_id)
    safe_filename, _ = _validate_filename(filename)
    source_documents = _extract_documents(safe_filename, content)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.KNOWLEDGE_CHUNK_SIZE,
        chunk_overlap=settings.KNOWLEDGE_CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(source_documents)
    if not chunks:
        raise KnowledgeBaseError(f"No text chunks were produced for {safe_filename}.")

    document_id = uuid4().hex
    uploaded_at = datetime.now(UTC).isoformat()
    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "knowledge_base_id": safe_id,
                "document_id": document_id,
                "source": safe_filename,
                "content_type": content_type or "application/octet-stream",
                "chunk_index": chunk_index,
                "uploaded_at": uploaded_at,
            }
        )

    vector_store = _vector_store(safe_id)
    vector_store.add_documents(
        chunks,
        ids=[f"{document_id}:{chunk_index}" for chunk_index in range(len(chunks))],
    )
    return IngestResult(
        knowledge_base_id=safe_id,
        document_id=document_id,
        filename=safe_filename,
        content_type=content_type or "application/octet-stream",
        size_bytes=len(content),
        chunk_count=len(chunks),
    )


def search_documents(
    knowledge_base_id: str,
    query: str,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Return the most relevant chunks with source metadata for citations."""
    if not query.strip():
        raise KnowledgeBaseError("Search query cannot be empty.")
    if top_k is not None and not 1 <= top_k <= 20:
        raise KnowledgeBaseError("top_k must be between 1 and 20.")

    vector_store = _vector_store(knowledge_base_id)
    matches = vector_store.similarity_search_with_score(
        query,
        k=top_k or settings.KNOWLEDGE_DEFAULT_TOP_K,
    )
    results: list[SearchResult] = []
    for document, score in matches:
        metadata = dict(document.metadata)
        page = metadata.get("page")
        results.append(
            SearchResult(
                document_id=str(metadata.get("document_id", "")),
                source=str(metadata.get("source", "unknown")),
                page=int(page) if isinstance(page, (int, float)) else None,
                content=document.page_content,
                score=float(score),
                metadata=metadata,
            )
        )
    return results
