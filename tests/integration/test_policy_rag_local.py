"""Opt-in semantic retrieval checks using cached weights and real Chroma."""

import pytest

from agents.local_embeddings import EMBEDDING_ID, LocalBGEEmbeddings
from agents.policy_rag import PROJECT_ROOT, PolicyRAG
from core.settings import settings

pytestmark = pytest.mark.local_rag


@pytest.fixture(scope="module")
def real_policy_rag(tmp_path_factory):
    return PolicyRAG(
        knowledge_dir=PROJECT_ROOT / "knowledge/after_sales",
        persist_dir=tmp_path_factory.mktemp("real-policy-chroma"),
        embeddings=LocalBGEEmbeddings(PROJECT_ROOT / settings.POLICY_LOCAL_CACHE_DIR),
        embedding_id=EMBEDDING_ID,
        min_score=settings.POLICY_MIN_SCORE,
        top_k=settings.POLICY_TOP_K,
        chunk_size=400,
        chunk_overlap=60,
    )


@pytest.mark.parametrize(
    "question,source,clause",
    [
        ("质量问题退款需要什么材料？", "refund_policy.md", "照片或视频"),
        ("退款需要人工审批吗？", "refund_policy.md", "人工审批"),
        ("怎么申请换货，需要哪些材料？", "return_exchange_policy.md", "商品照片"),
        ("退货运费谁承担？", "return_exchange_policy.md", "运费争议"),
        ("包裹物流延迟怎么办？", "shipping_policy.md", "运单号"),
        ("物流显示签收但没有收到怎么办？", "shipping_policy.md", "签收但未收到"),
    ],
)
def test_policy_question_returns_relevant_clause(real_policy_rag, question, source, clause):
    result = real_policy_rag.search(question)
    assert result["status"] == "ok", result
    assert any(
        item["source"] == source and clause in item["content"] for item in result["evidence"]
    ), result


@pytest.mark.parametrize(
    "question",
    [
        "明天天气怎么样？",
        "今天足球比赛谁赢了？",
        "如何用Python写一个循环？",
        "怎样修改账号密码？",
    ],
)
def test_unrelated_question_returns_no_policy(real_policy_rag, question):
    result = real_policy_rag.search(question)
    assert result["status"] == "no_evidence", result
    assert result["evidence"] == []
