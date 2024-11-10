import uvicorn
from dotenv import load_dotenv

from core import settings

load_dotenv()

if __name__ == "__main__":
    if not settings.is_dev():
        uvicorn.run("service:app", host=settings.HOST, port=settings.PORT)
    else:
        uvicorn.run("service:app", reload=True)
