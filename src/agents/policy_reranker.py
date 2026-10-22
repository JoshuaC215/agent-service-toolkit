"""Adapted from HelpPilot's retrieve/query-passage reranking pipeline (MIT).

Copyright (c) 2026 <Your Name> (upstream notice).
See third_party/helppilot/LICENSE and THIRD_PARTY_NOTICES.md.
Chinese model, explicit download, validation and offline loading are local additions.
"""

import json
import math
from pathlib import Path

MODEL_NAME = "BAAI/bge-reranker-base"


class LocalPolicyReranker:
    def __init__(self, cache_dir: Path):
        from sentence_transformers import CrossEncoder

        root = cache_dir.resolve()
        manifest = json.loads((root / "model.json").read_text(encoding="utf-8"))
        snapshot = (root / manifest["snapshot"]).resolve()
        if manifest["model_id"] != MODEL_NAME or not snapshot.is_relative_to(root):
            raise ValueError("Invalid reranker manifest")
        if snapshot.name != manifest["revision"]:
            raise ValueError("Reranker revision does not match snapshot")
        self.model = CrossEncoder(
            str(snapshot),
            device="cpu",
            max_length=512,
            local_files_only=True,
            trust_remote_code=False,
        )

    def rank(self, question: str, evidence: list[dict]) -> list[dict]:
        if not evidence:
            return []
        # Adapted from helppilot/rag.py: predict on (query, passage) pairs, sort
        # together with metadata, then select top-k in the caller.
        scores = self.model.predict([(question, item["content"]) for item in evidence])
        ranked = []
        for score, item in zip(scores, evidence, strict=True):
            value = float(score)
            if not math.isfinite(value):
                raise ValueError("Non-finite rerank score")
            ranked.append({**item, "rerank_score": round(value, 6)})
        return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)
