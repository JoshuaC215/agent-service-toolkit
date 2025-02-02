from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from src.agents.similarity_agent import SimilarityAgent  


class AgentState(MessagesState, total=False):
    """Estado do agente, armazenando as mensagens da conversa."""


async def process_query(state: AgentState, config: RunnableConfig) -> AgentState:
    """Processa a consulta utilizando o agente de similaridade para buscar a melhor resposta."""
    similarity_agent = SimilarityAgent()
    
    # Obtém a melhor resposta entre a base interna e a API do GPT
    best_response = similarity_agent.get_best_response(state["messages"][-1].content)

    response = AIMessage(
        content=(
            f"\n **Categoria**: {best_response['categoria']}\n"
            f" **Pergunta**: {best_response['pergunta']}\n"
            f" **Resposta**: {best_response['resposta']}\n"
            f" **Contexto**: {best_response['contexto']}\n"
        )
    )

    return {"messages": [response]}


# Define o fluxo do chatbot
agent = StateGraph(AgentState)
agent.add_node("process_query", process_query)
agent.set_entry_point("process_query")

# Finaliza após o processamento da resposta
agent.add_edge("process_query", END)

chatbot = agent.compile(
    checkpointer=MemorySaver(),
)
