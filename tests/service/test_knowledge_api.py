from unittest.mock import patch

from core import settings
from knowledge import IngestResult, KnowledgeBaseError, SearchResult


def test_upload_knowledge_document(test_client):
    ingest_result = IngestResult(
        knowledge_base_id="handbook",
        document_id="document-1",
        filename="policy.md",
        content_type="text/markdown",
        size_bytes=12,
        chunk_count=1,
    )
    with patch("service.service.ingest_document", return_value=ingest_result) as ingest:
        response = test_client.post(
            "/knowledge-bases/handbook/documents",
            files={"file": ("policy.md", b"hello policy", "text/markdown")},
        )

    assert response.status_code == 201
    assert response.json()["document_id"] == "document-1"
    ingest.assert_called_once_with("handbook", "policy.md", b"hello policy", "text/markdown")


def test_upload_knowledge_document_returns_validation_error(test_client):
    with patch(
        "service.service.ingest_document",
        side_effect=KnowledgeBaseError("Unsupported document type."),
    ):
        response = test_client.post(
            "/knowledge-bases/handbook/documents",
            files={"file": ("policy.exe", b"not a document", "application/octet-stream")},
        )

    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_search_knowledge_base_exposes_source_metadata(test_client):
    match = SearchResult(
        document_id="document-1",
        source="policy.md",
        page=1,
        content="Remote work policy",
        score=0.12,
        metadata={"chunk_index": 0},
    )
    with patch("service.service.search_documents", return_value=[match]) as search:
        response = test_client.get(
            "/knowledge-bases/handbook/search",
            params={"q": "remote work", "k": 3},
        )

    assert response.status_code == 200
    assert response.json()["results"][0]["source"] == "policy.md"
    assert response.json()["results"][0]["page"] == 1
    search.assert_called_once_with("handbook", "remote work", 3)


def test_upload_and_search_knowledge_base_end_to_end(test_client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "KNOWLEDGE_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "USE_FAKE_MODEL", True)

    upload_response = test_client.post(
        "/knowledge-bases/handbook/documents",
        files={
            "file": (
                "policy.md",
                b"Remote work requests must be submitted five business days in advance.",
                "text/markdown",
            )
        },
    )
    assert upload_response.status_code == 201
    assert upload_response.json()["chunk_count"] == 1

    search_response = test_client.get(
        "/knowledge-bases/handbook/search",
        params={"q": "remote work request", "k": 3},
    )
    assert search_response.status_code == 200
    assert search_response.json()["results"][0]["source"] == "policy.md"
