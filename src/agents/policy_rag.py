"""Persistent, versioned retrieval of the demo after-sales policies."""

import hashlib
import json
import math
from pathlib import Path
from threading import RLock

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.name == "src":
    PROJECT_ROOT = PROJECT_ROOT.parent


class PolicyRAGError(RuntimeError):
    """A policy index cannot safely supply evidence."""


def load_policy_documents(directory: Path) -> list[Document]:
    documents = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), "")
        if not title:
            raise PolicyRAGError(f"Policy document needs a title: {path.name}")
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "title": title, "demo_policy": True},
            )
        )
    if not documents:
        raise PolicyRAGError("No policy Markdown documents found")
    return documents


class PolicyRAG:
    def __init__(
        self,
        *,
        knowledge_dir: Path,
        persist_dir: Path,
        embeddings: Embeddings,
        embedding_id: str,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        min_score: float = 0.65,
        top_k: int = 3,
    ):
        if not embedding_id.strip():
            raise ValueError("embedding_id must identify the provider and embedding model")
        if not 0 <= min_score <= 1 or not 1 <= top_k <= 10:
            raise ValueError("Invalid retrieval score threshold or top_k")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.knowledge_dir = Path(knowledge_dir)
        self.persist_dir = Path(persist_dir)
        self.embeddings = embeddings
        self.embedding_id = embedding_id
        self.min_score = min_score
        self.top_k = top_k
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n\n", "\n", "。", "；", " ", ""],
        )
        self._chunk_config = (chunk_size, chunk_overlap)
        self._lock = RLock()
        self._client = None

    def _prepare_index(self) -> tuple[Chroma, int]:
        documents = load_policy_documents(self.knowledge_dir)
        snapshot = {
            "version": 1,
            "embedding": self.embedding_id,
            "chunk_config": self._chunk_config,
            "documents": [(doc.metadata, doc.page_content) for doc in documents],
        }
        fingerprint = hashlib.sha256(
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:32]
        chunks = self.splitter.split_documents(documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{fingerprint}:{index}"
        if self._client is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        store = Chroma(
            client=self._client,
            collection_name=f"after_sales_{fingerprint}",
            embedding_function=self.embeddings,
            collection_configuration={"hnsw": {"space": "cosine"}},
        )
        expected_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        # A partial build is retried; old policy generations are never searched.
        if set(store.get(include=[])["ids"]) != set(expected_ids):
            for start in range(0, len(chunks), 64):
                store.add_documents(
                    chunks[start : start + 64], ids=expected_ids[start : start + 64]
                )
        return store, len(chunks)

    def build_index(self) -> int:
        with self._lock:
            _, count = self._prepare_index()
            return count

    def search(self, question: str) -> dict:
        question = question.strip()
        if not question or len(question) > 2000:
            raise ValueError("Policy question must contain 1 to 2000 characters")
        with self._lock:
            store, count = self._prepare_index()
            candidates = store.similarity_search_with_score(question, k=min(self.top_k, count))
        evidence = []
        for document, distance in candidates:
            score = 1.0 - float(distance)
            if not math.isfinite(score) or score < self.min_score:
                continue
            evidence.append(
                {
                    "source": document.metadata["source"],
                    "title": document.metadata["title"],
                    "chunk_id": document.metadata["chunk_id"],
                    "content": document.page_content,
                    "score": round(min(1.0, max(0.0, score)), 4),
                }
            )
        return {
            "status": "ok" if evidence else "no_evidence",
            "found": bool(evidence),
            "demo_policy": True,
            "requires_human_approval": True,
            "evidence": evidence,
        }
