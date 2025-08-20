# Contributing to Agent Service Toolkit

Thank you for your interest in contributing to the Agent Service Toolkit! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/agent-service-toolkit.git
   cd agent-service-toolkit
   ```

2. **Set up your development environment**
   ```bash
   # Install uv (recommended package manager)
   curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
   
   # Install dependencies
   uv sync --frozen
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Create a branch for your changes**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📋 Ways to Contribute

### 🐛 Bug Reports
- Use the [GitHub Issues](https://github.com/JoshuaC215/agent-service-toolkit/issues) to report bugs
- Include a clear description of the problem
- Provide steps to reproduce the issue
- Include environment details (Python version, OS, etc.)

### 💡 Feature Requests
- Open a [GitHub Issue](https://github.com/JoshuaC215/agent-service-toolkit/issues) with the "enhancement" label
- Describe the feature and its use case
- Explain why it would benefit the project

### 🔧 Code Contributions
- Bug fixes
- New features
- Documentation improvements
- Test coverage improvements
- Performance optimizations

## 🏗️ Development Setup

### Prerequisites
- Python 3.11 or higher
- Docker and Docker Compose (>= v2.23.0) for containerized development
- Git

### Environment Setup
1. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys (at least one LLM provider required)
   ```

2. **Development with Docker (Recommended)**
   ```bash
   docker compose watch
   ```
   This provides automatic reloading during development.

3. **Local Development**
   ```bash
   # Start the FastAPI service
   python src/run_service.py
   
   # In another terminal, start Streamlit app
   streamlit run src/streamlit_app.py
   ```

### LangGraph Studio Development
The project supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) for agent development:

```bash
langgraph dev
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Writing Tests
- Write tests for all new functionality
- Maintain or improve code coverage
- Follow existing test patterns in the `tests/` directory
- Use descriptive test names and docstrings

### Test Structure
```
tests/
├── conftest.py              # Shared test fixtures
├── client/                  # Client tests
├── core/                    # Core module tests
├── service/                 # Service tests
├── app/                     # Streamlit app tests
└── integration/             # End-to-end tests
```

## 🏛️ Project Structure

Understanding the codebase structure:

```
src/
├── agents/                  # Agent implementations
│   ├── agents.py           # Agent registry
│   ├── chatbot.py          # Basic chatbot agent
│   ├── research_assistant.py # Research agent
│   └── rag_assistant.py    # RAG-enabled agent
├── client/                  # Client library
├── core/                    # Core modules (LLM, settings)
├── schema/                  # Data models and schemas
├── service/                 # FastAPI service
└── memory/                  # Memory/persistence layers
```

## 📝 Coding Standards

### Code Style
- **Formatter**: Project uses `ruff` for code formatting
- **Linter**: `ruff` for linting with additional rules
- **Type Checking**: `mypy` for static type analysis
- **Line Length**: 100 characters maximum

### Pre-commit Hooks
The project uses pre-commit hooks to ensure code quality:
```bash
pre-commit run --all-files  # Run all hooks manually
```

### Code Quality Commands
```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/
```

## 🤖 Adding New Agents

To create a new agent:

1. **Create agent file** in `src/agents/`
   ```python
   # src/agents/my_agent.py
   from langgraph import StateGraph
   from src.schema.schema import AgentState

   def create_my_agent() -> StateGraph:
       # Your agent implementation
       pass
   ```

2. **Register agent** in `src/agents/agents.py`
   ```python
   from .my_agent import create_my_agent

   agents = {
       "my_agent": create_my_agent(),
       # ... other agents
   }
   ```

3. **Add tests** in `tests/agents/`
4. **Update documentation** if needed

## 📖 Documentation

### Documentation Structure
- `README.md` - Main project documentation
- `docs/` - Detailed documentation for specific features
- Code comments and docstrings for complex logic

### Writing Documentation
- Use clear, concise language
- Include code examples where helpful
- Update relevant docs when adding features
- Follow existing documentation patterns

## 🔄 Pull Request Process

### Before Submitting
1. **Test your changes**
   ```bash
   pytest
   ruff check .
   mypy src/
   ```

2. **Update documentation** if needed

3. **Write descriptive commit messages**
   ```bash
   git commit -m "feat: add new research agent with web search capability"
   ```

### PR Guidelines
- **Title**: Use a descriptive title
- **Description**: Explain what changes you made and why
- **Link Issues**: Reference any related issues
- **Tests**: Ensure all tests pass
- **Documentation**: Update docs if needed

### PR Template
When creating a PR, please include:

```markdown
## Summary
Brief description of changes

## Changes Made
- List of specific changes
- Any breaking changes

## Testing
- [ ] All tests pass
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Updated relevant documentation
- [ ] Added code comments where needed
```

## 🔍 Code Review Process

### For Contributors
- Be open to feedback and suggestions
- Respond to review comments promptly
- Make requested changes in additional commits
- Squash commits before final merge if requested

### Review Criteria
- Code functionality and correctness
- Test coverage and quality
- Documentation completeness
- Code style adherence
- Performance considerations

## 🌟 Recognition

Contributors are recognized in several ways:
- GitHub contributors list
- Mention in release notes for significant contributions
- Community acknowledgment in discussions

## 📞 Getting Help

### Community Support
- **GitHub Discussions**: For questions and community interaction
- **Issues**: For bug reports and feature requests
- **Discord/Slack**: Check README for community chat links

### Maintainer Contact
For urgent matters or sensitive issues, contact the maintainers directly through GitHub.

## 📜 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

**Happy Contributing! 🎉**

Your contributions help make the Agent Service Toolkit better for everyone. Thank you for taking the time to contribute!