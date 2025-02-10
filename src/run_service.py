import asyncio
import sys

import uvicorn
from dotenv import load_dotenv

from core import settings

load_dotenv()

if __name__ == "__main__":
    # Set Compatible event loop policy on Windows Systems.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(
        "service:app", host=settings.HOST, port=settings.PORT, reload=settings.is_dev()
    )
