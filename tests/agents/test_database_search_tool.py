from types import SimpleNamespace

import pytest

import agents.tools as tools
from agents.tools import database_search_func, format_contexts


def test_format_contexts_joins_with_blank_lines():
    docs = [
        SimpleNamespace(page_content="First doc line\nwith more"),
        SimpleNamespace(page_content="Second doc content"),
    ]
    out = format_contexts(docs)
    assert out == "First doc line\nwith more\n\nSecond doc content"


@pytest.mark.asyncio
async def test_database_search_returns_formatted_context_and_preserves_links(monkeypatch):
    # Build fake retriever returning objects with .page_content containing a markdown link
    docs = [
        SimpleNamespace(page_content="Policy reference [Doc A](https://example.com/a)"),
        SimpleNamespace(page_content="Additional details [Doc B](https://example.com/b)"),
    ]

    class FakeRetriever:
        def invoke(self, query: str):
            # Ensure the query is passed but we don't depend on it
            assert isinstance(query, str)
            return docs

    # Monkeypatch the DB loader to avoid real embeddings/IO
    monkeypatch.setattr(tools, "load_chroma_db", lambda: FakeRetriever())

    out = database_search_func("any query")
    # Expect the same formatting as format_contexts
    assert out == format_contexts(docs)
    # Links (citations) should be preserved verbatim for the model to cite later
    assert "https://example.com/a" in out and "https://example.com/b" in out
