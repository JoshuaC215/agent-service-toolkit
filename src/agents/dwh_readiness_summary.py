from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core import settings
from core.llm import get_model  # zentrale Modell-Fabrik

"""
LangGraph-Agent für den DWH-Readiness-Check.

Graph-Idee (minimal):

1. format_input – nimmt das von Streamlit übergebene readiness_result-Dict
   (answers, slider, infos, scores, timestamp) und baut daraus einen
   menschenlesbaren Prompt.

2. llm_summary – LLM-Node (GPT-4o) erzeugt Zusammenfassung + 3 Empfehlungen
   im JSON-Format:
      { "summary": "...", "recommendations": ["...", "...", "..."] }

Der Agent wird in `src/agents/agents.py` unter dem Key
"dwh_readiness_summary" registriert.
"""


# ------- 1. State ------------------------------------------------------------


class AgentState(MessagesState, total=False):
    """State für DWH Readiness: erweitert MessagesState um Eingabe/Ausgabe."""

    readiness_result: dict[str, Any]
    llm_output: dict[str, Any] | None


# ------- 2. Nodes ------------------------------------------------------------


def format_input(state: AgentState) -> AgentState:
    """Pass-through: stelle sicher, dass der MessagesState-Key gesetzt bleibt (mypy-kompatibel)."""
    return {"messages": state.get("messages", [])}  # type: ignore[return-value]


def llm_summary(state: AgentState, config: RunnableConfig) -> AgentState:
    """Ruft LLM auf (modell-/token-gesteuert via config), parst JSON und liefert eine AIMessage."""
    # Prompt direkt aus readiness_result erzeugen (keine Zusatzkeys im State)
    data = state.get("readiness_result", {})
    prompt_text = (
        "Du bist ein Data-Warehouse-Experte.\n"
        "Fasse die folgenden Eingaben prägnant zusammen (max. 120 Wörter) "
        "und gib anschließend drei konkrete Handlungsempfehlungen (nummeriert) aus.\n\n"
        f"Eingaben:\n{data}\n\n"
        "Antwortformat (JSON ausschließlich):\n"
        '{ "summary": "...", "recommendations": ["...", "...", "..."] }'
    )

    cfg = config.get("configurable", {}) if hasattr(config, "get") else {}
    model_name = cfg.get("model", settings.DEFAULT_MODEL)
    owui_token = cfg.get("owui_token") or cfg.get("api_key") or ""
    model = get_model(model_name, owui_token).with_config(tags=["skip_stream"])

    prompt = ChatPromptTemplate.from_template("{prompt}")
    chain = prompt | model
    response = chain.invoke({"prompt": prompt_text})

    json_text = str(response.content)
    try:
        parsed = json.loads(json_text)
    except Exception:  # pragma: no cover
        parsed = {"summary": json_text, "recommendations": []}

    return {
        "messages": [AIMessage(content=json.dumps(parsed, ensure_ascii=False))],
        "llm_output": parsed,
    }  # type: ignore[return-value]


# ------- 3. Graph-Definition -------------------------------------------------


def dwh_readiness_summary_agent() -> CompiledStateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("format_input", format_input)
    graph.add_node("llm_summary", llm_summary)

    # Pfad: start -> format_input -> llm_summary -> END
    graph.set_entry_point("format_input")
    graph.add_edge("format_input", "llm_summary")
    graph.add_edge("llm_summary", END)
    return graph.compile()


# --------------------------------------------------------------------------- #
# Export compiled graph so that the service can register the agent easily
# Variable-Name-Konvention wie in den anderen Agent-Modulen
dwh_readiness_summary = dwh_readiness_summary_agent()
