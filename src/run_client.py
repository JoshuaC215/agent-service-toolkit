import asyncio
import logging

from client import AgentClient
from core import settings
from core.logging import setup_logging
from schema import ChatMessage

logger = logging.getLogger(__name__)
setup_logging()


async def amain() -> None:
    #### ASYNC ####
    client = AgentClient(settings.BASE_URL)

    logger.info("Agent info:")
    logger.info("%s", client.info)

    logger.info("Chat example:")
    response = await client.ainvoke("Tell me a brief joke?", model="gpt-4o")
    response.pretty_print()

    logger.info("Stream example:")
    async for message in client.astream("Share a quick fun fact?"):
        if isinstance(message, str):
            logger.info("%s", message)
        elif isinstance(message, ChatMessage):
            message.pretty_print()
        else:
            logger.error("Unknown type - %s", type(message))


def main() -> None:
    #### SYNC ####
    client = AgentClient(settings.BASE_URL)

    logger.info("Agent info:")
    logger.info("%s", client.info)

    logger.info("Chat example:")
    response = client.invoke("Tell me a brief joke?", model="gpt-4o")
    response.pretty_print()

    logger.info("Stream example:")
    for message in client.stream("Share a quick fun fact?"):
        if isinstance(message, str):
            logger.info("%s", message)
        elif isinstance(message, ChatMessage):
            message.pretty_print()
        else:
            logger.error("Unknown type - %s", type(message))


if __name__ == "__main__":
    logger.info("Running in sync mode")
    main()
    logger.info("Running in async mode")
    asyncio.run(amain())
