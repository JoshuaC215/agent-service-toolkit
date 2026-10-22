import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.policy_reranker import MODEL_NAME, LocalPolicyReranker


def test_cross_encoder_scores_question_passage_pairs_and_preserves_metadata():
    ranker = LocalPolicyReranker.__new__(LocalPolicyReranker)
    ranker.model = Mock()
    ranker.model.predict.return_value = [0.1, 0.9]
    evidence = [
        {"content": "A", "source": "a.md", "chunk_id": "a", "score": 0.7},
        {"content": "B", "source": "b.md", "chunk_id": "b", "score": 0.6},
    ]
    result = ranker.rank("question", evidence)
    ranker.model.predict.assert_called_once_with([("question", "A"), ("question", "B")])
    assert [item["chunk_id"] for item in result] == ["b", "a"]
    assert result[0]["score"] == 0.6
    assert result[0]["rerank_score"] == 0.9
    assert "rerank_score" not in evidence[0]


@pytest.mark.parametrize("scores", [[float("nan")], []])
def test_invalid_rerank_scores_fail_closed(scores):
    ranker = LocalPolicyReranker.__new__(LocalPolicyReranker)
    ranker.model = Mock()
    ranker.model.predict.return_value = scores
    with pytest.raises(ValueError):
        ranker.rank("question", [{"content": "text"}])


def test_loader_uses_only_local_snapshot_and_cpu(tmp_path, monkeypatch):
    constructor = Mock()
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=constructor)
    )
    revision = "a" * 40
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (tmp_path / "model.json").write_text(
        json.dumps(
            {"model_id": MODEL_NAME, "revision": revision, "snapshot": f"snapshots/{revision}"}
        ),
        encoding="utf-8",
    )
    LocalPolicyReranker(tmp_path)
    constructor.assert_called_once_with(
        str(snapshot.resolve()),
        device="cpu",
        max_length=512,
        local_files_only=True,
        trust_remote_code=False,
    )


def test_loader_rejects_snapshot_outside_cache(tmp_path, monkeypatch):
    constructor = Mock()
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=constructor)
    )
    (tmp_path / "model.json").write_text(
        json.dumps({"model_id": MODEL_NAME, "revision": "bad", "snapshot": "../bad"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid reranker manifest"):
        LocalPolicyReranker(tmp_path)
    constructor.assert_not_called()
