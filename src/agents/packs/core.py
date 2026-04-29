from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.command_agent import command_agent
from agents.interrupt_agent import interrupt_agent
from agents.knowledge_base_agent import kb_agent
from agents.langgraph_supervisor_agent import langgraph_supervisor_agent
from agents.rag_assistant import rag_assistant
from agents.research_assistant import research_assistant
from agents.packs.types import AgentSpec


def get_core_agent_specs() -> list[AgentSpec]:
    return [
        AgentSpec(
            key="chatbot",
            description="A simple chatbot.",
            graph=chatbot,
            track="core",
            stability="stable",
            pack="core",
        ),
        AgentSpec(
            key="research-assistant",
            description="A research assistant with web search and calculator.",
            graph=research_assistant,
            track="core",
            stability="stable",
            pack="core",
        ),
        AgentSpec(
            key="rag-assistant",
            description="A RAG assistant with access to information in a database.",
            graph=rag_assistant,
            track="core",
            stability="beta",
            pack="core",
        ),
        AgentSpec(
            key="command-agent",
            description="A command agent.",
            graph=command_agent,
            track="core",
            stability="beta",
            pack="core",
        ),
        AgentSpec(
            key="bg-task-agent",
            description="A background task agent.",
            graph=bg_task_agent,
            track="core",
            stability="beta",
            pack="core",
        ),
        AgentSpec(
            key="langgraph-supervisor-agent",
            description="A langgraph supervisor agent.",
            graph=langgraph_supervisor_agent,
            track="core",
            stability="beta",
            pack="core",
        ),
        AgentSpec(
            key="interrupt-agent",
            description="An agent that uses interrupts.",
            graph=interrupt_agent,
            track="core",
            stability="stable",
            pack="core",
        ),
        AgentSpec(
            key="knowledge-base-agent",
            description="A retrieval-augmented generation agent using Amazon Bedrock Knowledge Base.",
            graph=kb_agent,
            track="core",
            stability="beta",
            pack="core",
        ),
    ]
