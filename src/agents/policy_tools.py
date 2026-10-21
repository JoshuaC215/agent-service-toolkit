"""Agent tool and explicit embedding configuration for policy retrieval."""

import logging
from functools import cache
from pathlib import Path

from langchain_core.tools import tool

from agents.policy_rag import PROJECT_ROOT, PolicyRAG, PolicyRAGError

logger = logging.getLogger(__name__)


@cache
def get_policy_rag() -> PolicyRAG:
    from core.settings import settings

    provider = settings.POLICY_EMBEDDING_PROVIDER
    if provider == "local":
        from agents.local_embeddings import EMBEDDING_ID, MODEL_NAME, LocalBGEEmbeddings

        if settings.POLICY_EMBEDDING_MODEL not in (None, MODEL_NAME):
            raise PolicyRAGError(f"The local adapter supports {MODEL_NAME}")
        embeddings = LocalBGEEmbeddings(PROJECT_ROOT / settings.POLICY_LOCAL_CACHE_DIR)
        embedding_id = EMBEDDING_ID
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.OPENAI_API_KEY:
            raise PolicyRAGError("Configure OPENAI_API_KEY for policy embeddings")
        model = settings.POLICY_EMBEDDING_MODEL or "text-embedding-3-small"
        embeddings = OpenAIEmbeddings(
            model=model,
            api_key=settings.OPENAI_API_KEY,
            timeout=20,
            max_retries=1,
        )
        embedding_id = f"{provider}:{model}"
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not settings.GOOGLE_API_KEY:
            raise PolicyRAGError("Configure GOOGLE_API_KEY for policy embeddings")
        model = settings.POLICY_EMBEDDING_MODEL or "gemini-embedding-001"
        embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=settings.GOOGLE_API_KEY,
        )
        embedding_id = f"{provider}:{model}"
    return PolicyRAG(
        knowledge_dir=PROJECT_ROOT / Path(settings.POLICY_KNOWLEDGE_DIR),
        persist_dir=PROJECT_ROOT / Path(settings.POLICY_PERSIST_DIR),
        embeddings=embeddings,
        embedding_id=embedding_id,
        min_score=settings.POLICY_MIN_SCORE,
        top_k=settings.POLICY_TOP_K,
        chunk_size=400 if provider == "local" else 600,
        chunk_overlap=60 if provider == "local" else 80,
    )


@tool
def search_after_sales_policy(question: str) -> dict:
    """Retrieve demo refund, return/exchange or shipping policy evidence with sources.

    Retrieval cannot approve or execute refunds. Empty evidence requires human verification.
    """
    if not question.strip() or len(question.strip()) > 2000:
        return {"status": "invalid_query", "found": False, "evidence": []}
    try:
        return get_policy_rag().search(question)
    except Exception as exc:
        logger.warning(
            "Policy retrieval unavailable (%s); no evidence supplied", type(exc).__name__
        )
        return {
            "status": "unavailable",
            "found": False,
            "demo_policy": True,
            "requires_human_approval": True,
            "evidence": [],
            "message": "政策检索暂不可用，请人工核实；不能据此承诺退款或编造政策。",
        }
