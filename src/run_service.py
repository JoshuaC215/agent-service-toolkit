import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    if os.getenv("MODE") != "dev":
        from service import app

        uvicorn.run(app, host="0.0.0.0", port=80)
    else:
        uvicorn.run("service:app", reload=True)
