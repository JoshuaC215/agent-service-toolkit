from dotenv import load_dotenv
import uvicorn

from service import app

load_dotenv()
uvicorn.run(app, host="0.0.0.0", port=80)
