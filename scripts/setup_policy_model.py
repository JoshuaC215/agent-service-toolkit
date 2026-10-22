"""Explicitly download the local policy model, then check its output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.local_embeddings import MODEL_NAME, MODEL_REVISION, LocalBGEEmbeddings  # noqa: E402
from agents.policy_rag import PROJECT_ROOT  # noqa: E402
from core.settings import settings  # noqa: E402


def main() -> None:
    if settings.POLICY_EMBEDDING_PROVIDER != "local":
        raise SystemExit("Set POLICY_EMBEDDING_PROVIDER=local before downloading")
    if settings.POLICY_EMBEDDING_MODEL not in (None, MODEL_NAME):
        raise SystemExit(f"The local adapter supports {MODEL_NAME}")
    print(f"Downloading {MODEL_NAME} at {MODEL_REVISION}; no embedding API is called.", flush=True)
    embeddings = LocalBGEEmbeddings(
        PROJECT_ROOT / settings.POLICY_LOCAL_CACHE_DIR, allow_download=True
    )
    vector = embeddings.embed_query("质量问题退款需要什么材料？")
    if len(vector) != 512:
        raise RuntimeError(f"Unexpected embedding dimension: {len(vector)}")
    print(f"Local CPU embedding ready: {len(vector)} dimensions.")


if __name__ == "__main__":
    main()
