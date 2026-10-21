"""CPU BGE embeddings with a pinned model and explicit first-time download."""

from pathlib import Path

from langchain_core.embeddings import Embeddings

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
EMBEDDING_ID = f"local:{MODEL_NAME}@{MODEL_REVISION}:normalized:zh-query-v1"


class LocalBGEEmbeddings(Embeddings):
    def __init__(self, cache_dir: Path, *, allow_download: bool = False):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[missing-import]
        except ImportError as exc:
            raise RuntimeError(
                "Install local embedding dependencies with: uv sync --extra local-rag"
            ) from exc
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(
            MODEL_NAME,
            revision=MODEL_REVISION,
            device="cpu",
            cache_folder=str(cache_dir),
            local_files_only=not allow_download,
            trust_remote_code=False,
        )
        self.model.max_seq_length = 512

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            prompt="",
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            [QUERY_INSTRUCTION + text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            prompt="",
        )[0].tolist()
