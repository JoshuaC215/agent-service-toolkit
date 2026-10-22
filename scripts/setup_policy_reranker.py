"""Explicit first-time download; normal retrieval never downloads model files."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.policy_rag import PROJECT_ROOT  # noqa: E402
from agents.policy_reranker import MODEL_NAME, LocalPolicyReranker  # noqa: E402
from core.settings import settings  # noqa: E402


def main():
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_NAME}; local CPU inference, no embedding API is called.", flush=True)
    cache = (PROJECT_ROOT / settings.POLICY_RERANK_CACHE_DIR).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            MODEL_NAME,
            cache_dir=str(cache),
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
        )
    ).resolve()
    manifest = {
        "model_id": MODEL_NAME,
        "revision": snapshot.name,
        "snapshot": snapshot.relative_to(cache).as_posix(),
    }
    pending = cache / "model.pending.json"
    pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Only publish the ready marker after a real inference succeeds.
    from sentence_transformers import CrossEncoder

    ranker = LocalPolicyReranker.__new__(LocalPolicyReranker)
    ranker.model = CrossEncoder(
        str(snapshot), device="cpu", max_length=512, local_files_only=True, trust_remote_code=False
    )
    ranker.rank("退款材料", [{"content": "需提供订单号和照片。"}])
    pending.replace(cache / "model.json")
    print(f"Local CPU reranker ready: {MODEL_NAME}@{snapshot.name}")


if __name__ == "__main__":
    main()
