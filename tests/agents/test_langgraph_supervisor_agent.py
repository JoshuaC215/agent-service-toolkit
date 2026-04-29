from agents.langgraph_supervisor_agent import web_search


def test_web_search_empty_query() -> None:
    assert web_search("   ") == "No query provided."


def test_web_search_formats_results(monkeypatch) -> None:
    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query: str, max_results: int = 5):
            assert query == "python testing"
            assert max_results == 5
            return [
                {
                    "title": "Pytest docs",
                    "href": "https://docs.pytest.org",
                    "body": "Good testing practices.",
                },
                {
                    "title": "Example",
                    "href": "https://example.com",
                    "body": "",
                },
            ]

    monkeypatch.setattr("agents.langgraph_supervisor_agent.DDGS", _FakeDDGS)

    result = web_search("python testing")
    assert "1. Pytest docs" in result
    assert "URL: https://docs.pytest.org" in result
    assert "Snippet: Good testing practices." in result
    assert "2. Example" in result


def test_web_search_failure(monkeypatch) -> None:
    class _BrokenDDGS:
        def __enter__(self):
            raise RuntimeError("boom")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("agents.langgraph_supervisor_agent.DDGS", _BrokenDDGS)

    result = web_search("python")
    assert result.startswith("Web search failed:")
