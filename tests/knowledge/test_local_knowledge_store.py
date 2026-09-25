import pytest

from core import settings
from knowledge import KnowledgeBaseError, ingest_document, search_documents


@pytest.fixture(autouse=True)
def configure_local_knowledge_base(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "KNOWLEDGE_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "USE_FAKE_MODEL", True)
    monkeypatch.setattr(settings, "KNOWLEDGE_CHUNK_SIZE", 80)
    monkeypatch.setattr(settings, "KNOWLEDGE_CHUNK_OVERLAP", 10)


def test_ingest_and_search_document_returns_citation_metadata():
    result = ingest_document(
        "handbook",
        "remote-work.md",
        b"Remote work requests must be submitted five business days in advance.",
        "text/markdown",
    )

    matches = search_documents("handbook", "How far in advance is remote work requested?", 3)

    assert result.knowledge_base_id == "handbook"
    assert result.filename == "remote-work.md"
    assert result.chunk_count == 1
    assert matches
    assert matches[0].document_id == result.document_id
    assert matches[0].source == "remote-work.md"
    assert matches[0].page == 1


def test_ingest_splits_long_document_and_keeps_document_identity():
    content = ("A handbook paragraph about safety procedures. " * 30).encode()

    result = ingest_document("handbook", "safety.txt", content)

    assert result.chunk_count > 1
    matches = search_documents("handbook", "safety procedures", 20)
    assert matches
    assert {match.document_id for match in matches} == {result.document_id}


def test_unsupported_extension_is_rejected():
    with pytest.raises(KnowledgeBaseError, match="Unsupported document type"):
        ingest_document("handbook", "script.py", b"print('no')")
