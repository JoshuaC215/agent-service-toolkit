import os

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# NOTE: models with streaming=True will send tokens as they are generated
# if the /stream endpoint is called with stream_tokens=True (the default)
models: dict[str, BaseChatModel] = {}
if os.getenv("OPENAI_API_KEY") is not None:
    models["gpt-4o-mini"] = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True)
if os.getenv("GROQ_API_KEY") is not None:
    models["llama-3.1-70b"] = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)
if os.getenv("GOOGLE_API_KEY") is not None:
    models["gemini-1.5-flash"] = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0.5, streaming=True
    )
if os.getenv("ANTHROPIC_API_KEY") is not None:
    models["claude-3-haiku"] = ChatAnthropic(
        model="claude-3-haiku-20240307", temperature=0.5, streaming=True
    )
if os.getenv("USE_AWS_BEDROCK") == "true":
    models["bedrock-haiku"] = ChatBedrock(
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0", temperature=0.5
    )

if not models:
    print("No LLM available. Please set environment variables to enable at least one LLM.")
    if os.getenv("MODE") == "dev":
        print("FastAPI initialization failed. Please use Ctrl + C to exit uvicorn.")
    exit(1)
