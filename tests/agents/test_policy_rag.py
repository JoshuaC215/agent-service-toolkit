"""Use real local Chroma with deterministic test vectors, never external APIs."""

from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings

from agents.policy_rag import PROJECT_ROOT, PolicyRAG, PolicyRAGError, load_policy_documents
from agents.policy_tools import search_after_sales_policy


class TopicEmbeddings(Embeddings):
    def __init__(self):
        self.document_calls = 0
        self.fail_documents = False

    def _embed(self, text):
        groups = (("退款", "质量"), ("退换", "换货"), ("物流", "包裹"), ("天气", "足球"))
        vector = [float(any(word in text for word in group)) for group in groups]
        return vector if any(vector) else [0.0, 0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        self.document_calls += 1
        if self.fail_documents:
            raise RuntimeError("Simulated embedding outage")
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def policies(tmp_path):
    directory = tmp_path / "knowledge"
    directory.mkdir()
    texts = {
        "refund_policy.md": "# 退款政策（演示）\n质量问题退款需要照片并由人工审批。",
        "return_exchange_policy.md": "# 退换政策（演示）\n换货需要核实库存。",
        "shipping_policy.md": "# 物流政策（演示）\n包裹延误需要核实物流信息。",
    }
    for name, text in texts.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


def make_rag(policies, tmp_path, embeddings, **kwargs):
    return PolicyRAG(
        knowledge_dir=policies,
        persist_dir=tmp_path / "chroma",
        embeddings=embeddings,
        embedding_id=kwargs.pop("embedding_id", "test:topics-v1"),
        **kwargs,
    )


def test_project_policies_have_titles_and_sources():
    documents = load_policy_documents(PROJECT_ROOT / "knowledge/after_sales")
    assert {doc.metadata["source"] for doc in documents} == {
        "refund_policy.md",
        "return_exchange_policy.md",
        "shipping_policy.md",
    }
    assert all("演示" in doc.metadata["title"] for doc in documents)
    assert all("人工" in doc.page_content for doc in documents)


@pytest.mark.parametrize(
    "question,source",
    [
        ("质量问题退款需要什么？", "refund_policy.md"),
        ("换货怎么办？", "return_exchange_policy.md"),
        ("包裹物流延迟怎么办？", "shipping_policy.md"),
    ],
)
def test_retrieves_evidence_and_source(policies, tmp_path, question, source):
    rag = make_rag(policies, tmp_path, TopicEmbeddings())
    result = rag.search(question)
    assert result["status"] == "ok"
    assert result["demo_policy"] is True
    assert result["requires_human_approval"] is True
    assert result["evidence"][0]["source"] == source
    assert result["evidence"][0]["title"]
    assert result["evidence"][0]["chunk_id"]
    assert result["evidence"][0]["content"]
    assert result["evidence"][0]["score"] >= 0.65


def test_unrelated_question_returns_no_evidence(policies, tmp_path):
    result = make_rag(policies, tmp_path, TopicEmbeddings()).search("明天天气怎样？")
    assert result["status"] == "no_evidence"
    assert result["found"] is False
    assert result["evidence"] == []


def test_reuses_index_across_instances_and_queries(policies, tmp_path):
    first = TopicEmbeddings()
    rag = make_rag(policies, tmp_path, first)
    rag.search("退款")
    calls = first.document_calls
    rag.search("退款")
    assert first.document_calls == calls
    reopened = TopicEmbeddings()
    reopened.fail_documents = True
    assert make_rag(policies, tmp_path, reopened).search("退款")["found"] is True
    assert reopened.document_calls == 0


def test_updates_and_deleted_documents_do_not_return_stale_evidence(policies, tmp_path):
    embeddings = TopicEmbeddings()
    rag = make_rag(policies, tmp_path, embeddings)
    assert "照片" in rag.search("退款")["evidence"][0]["content"]
    calls = embeddings.document_calls
    (policies / "refund_policy.md").write_text(
        "# 退款政策（演示）\n退款材料现在需要视频证明，人工核实。",
        encoding="utf-8",
    )
    result = rag.search("退款")
    assert "视频" in result["evidence"][0]["content"]
    assert "照片" not in result["evidence"][0]["content"]
    assert embeddings.document_calls > calls
    (policies / "refund_policy.md").unlink()
    assert rag.search("退款")["evidence"] == []


def test_embedding_model_change_builds_separate_index(policies, tmp_path):
    make_rag(policies, tmp_path, TopicEmbeddings()).build_index()
    embeddings = TopicEmbeddings()
    make_rag(policies, tmp_path, embeddings, embedding_id="test:topics-v2").build_index()
    assert embeddings.document_calls > 0


def test_failed_build_can_be_retried(policies, tmp_path):
    embeddings = TopicEmbeddings()
    embeddings.fail_documents = True
    rag = make_rag(policies, tmp_path, embeddings)
    with pytest.raises(RuntimeError, match="Simulated embedding outage"):
        rag.search("退款")
    embeddings.fail_documents = False
    assert rag.search("退款")["found"] is True


def test_partial_build_is_completed_before_search(policies, tmp_path, monkeypatch):
    for index in range(65):
        (policies / f"extra_{index}.md").write_text(
            f"# 退款补充条款 {index}（演示）\n退款需要人工审批。", encoding="utf-8"
        )
    embeddings = TopicEmbeddings()
    original = embeddings.embed_documents
    calls = 0

    def fail_second_batch(texts):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Second batch failed")
        return original(texts)

    monkeypatch.setattr(embeddings, "embed_documents", fail_second_batch)
    rag = make_rag(policies, tmp_path, embeddings)
    with pytest.raises(RuntimeError, match="Second batch failed"):
        rag.build_index()
    monkeypatch.setattr(embeddings, "embed_documents", original)
    assert rag.build_index() == 68
    assert rag.search("退款")["found"] is True


@pytest.mark.parametrize(
    "provider,key_name", [("openai", "OPENAI_API_KEY"), ("google", "GOOGLE_API_KEY")]
)
def test_missing_provider_key_returns_unavailable(monkeypatch, provider, key_name):
    from agents.policy_tools import get_policy_rag
    from core.settings import settings

    get_policy_rag.cache_clear()
    monkeypatch.setattr(settings, "POLICY_EMBEDDING_PROVIDER", provider)
    monkeypatch.setattr(settings, key_name, None)
    try:
        result = search_after_sales_policy.invoke({"question": "退款需要什么材料？"})
        assert result["status"] == "unavailable"
        assert result["evidence"] == []
        with pytest.raises(PolicyRAGError, match=key_name):
            get_policy_rag()
    finally:
        get_policy_rag.cache_clear()


def test_empty_knowledge_does_not_reuse_old_index(policies, tmp_path):
    rag = make_rag(policies, tmp_path, TopicEmbeddings())
    rag.build_index()
    for path in policies.glob("*.md"):
        path.unlink()
    with pytest.raises(PolicyRAGError, match="No policy"):
        rag.search("退款")


@pytest.mark.parametrize("question", [" ", "x" * 2001])
def test_invalid_question_does_not_initialize_retrieval(monkeypatch, question):
    getter = Mock(side_effect=AssertionError("Must not initialize a provider"))
    monkeypatch.setattr("agents.policy_tools.get_policy_rag", getter)
    assert search_after_sales_policy.invoke({"question": question})["status"] == "invalid_query"
    getter.assert_not_called()


def test_tool_returns_evidence_and_fails_closed(monkeypatch, policies, tmp_path):
    rag = make_rag(policies, tmp_path, TopicEmbeddings())
    monkeypatch.setattr("agents.policy_tools.get_policy_rag", lambda: rag)
    assert search_after_sales_policy.invoke({"question": "退款"})["found"] is True
    monkeypatch.setattr(rag, "search", Mock(side_effect=RuntimeError("sensitive error")))
    result = search_after_sales_policy.invoke({"question": "退款"})
    assert result["status"] == "unavailable"
    assert result["evidence"] == []
    assert "sensitive error" not in str(result)


def test_existing_agent_tools_are_preserved():
    from agents.order_agent import instructions, tools

    assert {tool.name for tool in tools} == {
        "get_order",
        "create_ticket",
        "get_ticket",
        "list_tickets_by_order",
        "search_after_sales_policy",
    }
    assert "human verification" in instructions
    assert "Policy questions alone do not authorize" in instructions
