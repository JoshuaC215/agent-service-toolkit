# Complex Multitool Agent

## Overview

The **Complex Multitool Agent** is a feature that enhances this agent service with the ability to perform web searches and calculations. It provides real-time updates by showing results as they are processed.

## How to Run the Service

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- API keys for OpenAI

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/olenaliuby/agent-service-toolkit.git
   cd agent-service-toolkit
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   Follow the instructions in the main [README.md](README.md) to install dependencies using `uv` or another method specified there.

4. **Set Up Environment Variables**

   Create a `.env` file in the root directory with your API key:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

### Running the Service

1. **Start the Service**

   Run the following command to start the service:

   ```bash
   python src/run_service.py
   ```

2. **Access the User Interface**

   In another terminal, run:

   ```bash
   streamlit run src/streamlit_app.py
   ```

## Example Commands

### Using cURL

Make sure the service is running with `python src/run_service.py`, then to interact with the agent, use this `curl` command:

```bash
curl -X POST "http://localhost:8080/complex-multitool-agent/stream" \
-H "Content-Type: application/json" \
-d '{"input": "Find the prime minister of UK, then add 40 to their birth year"}'
```

## Testing

### Run Tests

To check if everything is working, run the test script:

```bash
pytest tests/test_complex_multitool.py
```

## Tool Dependencies

- **Web Search**: Uses DuckDuckGo to find information.
- **Calculator**: Performs math calculations.

## Partial Streaming

This feature lets you see results as they are being processed, giving you real-time updates.

## Future Code Style Improvements

For more consistent code style, I would like to add the following tools to the project:

- Add black formatter to the project
- Add isort to the project
- Add flake8 to the project

## Contact

For questions or support, please contact [olena.liuby@gmail.com](mailto:olena.liuby@gmail.com).
