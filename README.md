# 🧰 AI Agent Service Toolkit

This repository provides a blueprint and full toolkit for a LangGraph-based agent service architecture. It includes a [LangGraph](https://langchain-ai.github.io/langgraph/) agent, a [FastAPI](https://fastapi.tiangolo.com/) service to serve it, a client to interact with the service, and a [Streamlit](https://streamlit.io/) app that uses the client to provide a chat interface.

This project offers a template for you to easily build and run your own agents using the LangGraph framework. It demonstrates a complete setup from agent definition to user interface, making it easier to get started with LangGraph-based projects by providing a full, robust toolkit.

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=VqQti9nGoe4)**

## Overview

### [Try the app!](https://agent-service-toolkit.streamlit.app/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agent-service-toolkit.streamlit.app/)

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

### Quickstart

Run directly in python

```sh
# An OPENAI_API_KEY is required
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# uv is recommended but "pip install ." also works
pip install uv
uv sync --frozen
# "uv sync" creates .venv automatically
source .venv/bin/activate
python src/run_service.py

# In another shell
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

Run with docker

```sh
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

### Key Features

1. **LangGraph Agent**: A customizable agent built using the LangGraph framework.
1. **FastAPI Service**: Serves the agent with both streaming and non-streaming endpoints.
1. **Advanced Streaming**: A novel approach to support both token-based and message-based streaming.
1. **Content Moderation**: Implements LlamaGuard for content moderation (requires Groq API key).
1. **Streamlit Interface**: Provides a user-friendly chat interface for interacting with the agent.
1. **Multiple Agent Support**: Run multiple agents in the service and call by URL path
1. **Asynchronous Design**: Utilizes async/await for efficient handling of concurrent requests.
1. **Feedback Mechanism**: Includes a star-based feedback system integrated with LangSmith.
1. **Docker Support**: Includes Dockerfiles and a docker compose file for easy development and deployment.

### Key Files

The repository is structured as follows:

- `src/agents/research_assistant.py`: Defines the main LangGraph agent
- `src/agents/llama_guard.py`: Defines the LlamaGuard content moderation
- `src/agents/models.py`: Configures available models based on ENV
- `src/agents/agents.py`: Mapping of all agents provided by the service
- `src/schema/schema.py`: Defines the protocol schema
- `src/service/service.py`: FastAPI service to serve the agents
- `src/client/client.py`: Client to interact with the agent service
- `src/streamlit_app.py`: Streamlit app providing a chat interface

## Why LangGraph?

AI agents are increasingly being built with more explicitly structured and tightly controlled [Compound AI Systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/), with careful attention to the [cognitive architecture](https://blog.langchain.dev/what-is-a-cognitive-architecture/). At the time of this repo's creation, LangGraph seems like the most advanced open source framework for building such systems, with a high degree of control as well as support for features like concurrent execution, cycles in the graph, streaming results, built-in observability, and the rich ecosystem around LangChain.

I've spent a decent amount of time building with LangChain over the past year and experienced some of the commonly cited pain points. In building this out with LangGraph I found a few similar issues, but overall I like the direction and I'm happy with my choice to use it.

With that said, there are several other interesting projects in this space that are worth calling out, and I hope to spend more time building with them soon:

- [LlamaIndex Workflows](https://www.llamaindex.ai/blog/introducing-workflows-beta-a-new-way-to-create-complex-ai-applications-with-llamaindex) and [llama-agents](https://github.com/run-llama/llama-agents): LlamaIndex Workflows launched the day I started working on this. I've generally really liked the experience building with LlamaIndex and this looks very promising.
- [DSPy](https://github.com/stanfordnlp/dspy): The DSPy optimizer and approach also seems super interesting and promising. But the creator [has stated](https://github.com/stanfordnlp/dspy/issues/703#issuecomment-2016598529) they aren't focusing on agents yet. I will probably experiment with building some of the specific nodes in more complex agents using DSPy in the future.
- I know there are more springing up regularly, such as I recently came across [Prefect ControlFlow](https://github.com/PrefectHQ/ControlFlow).

## Setup and Usage

1. Clone the repository:

   ```sh
   git clone https://github.com/JoshuaC215/agent-service-toolkit.git
   cd agent-service-toolkit
   ```

2. Set up environment variables:
   Create a `.env` file in the root directory and add the following:

   ```sh
   # Provide at least one LLM API key to enable the agent service

   # Optional, to enable OpenAI gpt-4o-mini
   OPENAI_API_KEY=your_openai_api_key

   # Optional, to enable LlamaGuard and Llama 3.1
   GROQ_API_KEY=your_groq_api_key

   # Optional, to enable Gemini 1.5 Flash
   # See: https://ai.google.dev/gemini-api/docs/api-key
   GOOGLE_API_KEY=your_gemini_key

   # Optional, to enable Claude 3 Haiku
   # See: https://docs.anthropic.com/en/api/getting-started
   ANTHROPIC_API_KEY=your_anthropic_key

   # Optional, to enable AWS Bedrock models Haiku
   # See: https://docs.aws.amazon.com/bedrock/latest/userguide/setting-up.html
   USE_AWS_BEDROCK=true

   # Optional, to enable simple header-based auth on the service
   AUTH_SECRET=any_string_you_choose

   # Optional, to enable OpenWeatherMap
   OPENWEATHERMAP_API_KEY=your_openweathermap_api_key

   # Optional, to enable LangSmith tracing
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langchain_api_key
   LANGCHAIN_PROJECT=your_project

   # Optional, if MODE=dev, uvicorn will reload the server on file changes
   MODE=
   ```

3. You can now run the agent service and the Streamlit app locally, either with Docker or just using Python. The Docker setup is recommended for simpler environment setup and immediate reloading of the services when you make changes to your code.

### Docker Setup

This project includes a Docker setup for easy development and deployment. The `compose.yaml` file defines two services: `agent_service` and `streamlit_app`. The `Dockerfile` for each is in their respective directories.

For local development, we recommend using [docker compose watch](https://docs.docker.com/compose/file-watch/). This feature allows for a smoother development experience by automatically updating your containers when changes are detected in your source code.

1. Make sure you have Docker and Docker Compose (>=[2.23.0](https://docs.docker.com/compose/release-notes/#2230)) installed on your system.

2. Build and launch the services in watch mode:

   ```sh
   docker compose watch
   ```

3. The services will now automatically update when you make changes to your code:
   - Changes in the relevant python files and directories will trigger updates for the relevantservices.
   - NOTE: If you make changes to the `pyproject.toml` or `uv.lock` files, you will need to rebuild the services by running `docker compose up --build`.

4. Access the Streamlit app by navigating to `http://localhost:8501` in your web browser.

5. The agent service API will be available at `http://localhost:80`. You can also use the OpenAPI docs at `http://localhost:80/redoc`.

6. Use `docker compose down` to stop the services.

This setup allows you to develop and test your changes in real-time without manually restarting the services.

### Local development without Docker

You can also run the agent service and the Streamlit app locally without Docker, just using a Python virtual environment.

1. Create a virtual environment and install dependencies:

   ```sh
   pip install uv
   uv sync --frozen --extra dev
   source .venv/bin/activate
   ```

2. Run the FastAPI server:

   ```sh
   python src/run_service.py
   ```

3. In a separate terminal, run the Streamlit app:

   ```sh
   streamlit run src/streamlit_app.py
   ```

4. Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

### Development with LangGraph Studio

The agent supports [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio), a new IDE for developing agents in LangGraph.

You can simply install LangGraph Studio, add your `.env` file to the root directory as described above, and then launch LangGraph studio pointed at the root directory. Customize `langgraph.json` as needed.

### Contributing

Currently the tests need to be run using the local development without Docker setup. To run the tests for the agent service:

1. Ensure you're in the project root directory and have activated your virtual environment.

2. Install the development dependencies and pre-commit hooks:

   ```sh
   pip install uv
   uv sync --frozen --extra dev
   pre-commit install
   ```

3. Run the tests using pytest:

   ```sh
   pytest
   ```

## Customization

To customize the agent for your own use case:

1. Add your new agent to the `src/agents` directory. You can copy `research_assistant.py` or `chatbot.py` and modify it to change the agent's behavior and tools.
1. Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`. Your agent can be called by `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`.
1. Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.

## Building other apps on the AgentClient

The repo includes a generic `src/client/client.AgentClient` that can be used to interact with the agent service. This client is designed to be flexible and can be used to build other apps on top of the agent. It supports both synchronous and asynchronous invocations, and streaming and non-streaming requests.

See the `src/run_client.py` file for full examples of how to use the `AgentClient`. A quick example:

```python
from client import AgentClient
client = AgentClient()

response = client.invoke("Tell me a brief joke?")
response.pretty_print()
# ================================== Ai Message ==================================
#
# A man walked into a library and asked the librarian, "Do you have any books on Pavlov's dogs and Schrödinger's cat?"
# The librarian replied, "It rings a bell, but I'm not sure if it's here or not."

```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Roadmap

- [x] Get LlamaGuard working for content moderation (anyone know a reliable and fast hosted version?)
- [x] Add more sophisticated tools for the research assistant
- [x] Increase test coverage and add CI pipeline
- [x] Add support for multiple agents running on the same service, including non-chat agent
- [ ] Deployment instructions and configuration for cloud providers
- [ ] More ideas? File an issue or create a discussion!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
