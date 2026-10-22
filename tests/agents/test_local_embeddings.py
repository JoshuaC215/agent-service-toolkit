"""Test the adapter contract without downloading weights or importing PyTorch."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from agents.local_embeddings import (
    EMBEDDING_ID,
    MODEL_NAME,
    MODEL_REVISION,
    QUERY_INSTRUCTION,
    LocalBGEEmbeddings,
)


@pytest.fixture
def fake_transformer(monkeypatch):
    model = Mock()
    constructor = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=constructor,
        ),
    )
    return constructor, model


def test_model_is_cpu_only_and_cannot_download_during_requests(tmp_path, fake_transformer):
    constructor, model = fake_transformer
    LocalBGEEmbeddings(tmp_path)
    constructor.assert_called_once_with(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
        cache_folder=str(tmp_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    assert model.max_seq_length == 512


def test_explicit_setup_can_download(tmp_path, fake_transformer):
    constructor, _ = fake_transformer
    LocalBGEEmbeddings(tmp_path, allow_download=True)
    assert constructor.call_args.kwargs["local_files_only"] is False


def test_query_instruction_and_normalization(tmp_path, fake_transformer):
    _, model = fake_transformer
    model.encode.return_value = np.array([[0.6, 0.8]])
    embeddings = LocalBGEEmbeddings(tmp_path)
    assert embeddings.embed_query("退款条件") == [0.6, 0.8]
    assert model.encode.call_args.args[0] == [QUERY_INSTRUCTION + "退款条件"]
    assert model.encode.call_args.kwargs["normalize_embeddings"] is True


def test_documents_have_no_query_instruction(tmp_path, fake_transformer):
    _, model = fake_transformer
    model.encode.return_value = np.array([[0.6, 0.8]])
    embeddings = LocalBGEEmbeddings(tmp_path)
    assert embeddings.embed_documents(["退款政策"]) == [[0.6, 0.8]]
    assert model.encode.call_args.args[0] == ["退款政策"]
    assert model.encode.call_args.kwargs["normalize_embeddings"] is True
    model.encode.reset_mock()
    assert embeddings.embed_documents([]) == []
    model.encode.assert_not_called()


def test_factory_uses_local_model_without_cloud_keys(monkeypatch, tmp_path, fake_transformer):
    from agents.policy_tools import get_policy_rag
    from core.settings import settings

    get_policy_rag.cache_clear()
    monkeypatch.setattr(settings, "POLICY_EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(settings, "POLICY_EMBEDDING_MODEL", None)
    monkeypatch.setattr(settings, "POLICY_LOCAL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", None)
    try:
        rag = get_policy_rag()
        assert isinstance(rag.embeddings, LocalBGEEmbeddings)
        assert rag.embedding_id == EMBEDDING_ID
        assert get_policy_rag() is rag
        fake_transformer[0].assert_called_once()
    finally:
        get_policy_rag.cache_clear()


def test_missing_dependency_has_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="uv sync --extra local-rag"):
        LocalBGEEmbeddings(tmp_path)
