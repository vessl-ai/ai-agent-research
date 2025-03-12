# Deep Research Agent with CrewAI

A powerful deep research agent built with CrewAI that conducts comprehensive research on any topic using human-in-the-loop feedback and memory capabilities.

## Features

- 🔍 Web search using SerperDev API and Firecrawl
- 👤 Human-in-the-loop feedback integration
- 🧠 Memory functionality using Qdrant
- 📝 Structured research reports
- 🔄 Iterative improvement based on feedback
- 💬 Local chat interface with FastAPI backend
- 📚 Interactive API documentation
- 🛠️ Modern Python project management with uv
- 🕷️ Deep website analysis with Firecrawl

## Requirements

- Python 3.8+
- Rust (required for tiktoken dependency)
- Qdrant running locally or a cloud instance
- API keys for OpenAI and SerperDev

## Installation

1. Install Rust (required for tiktoken):
```bash
# On macOS and Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# On Windows
# Download and run rustup-init.exe from https://rustup.rs
```

2. Install uv (if not already installed):
```bash
pip install uv
```

3. Clone the repository:
```bash
git clone https://github.com/yourusername/crewai-deepresearch.git
cd crewai-deepresearch
```

3. Create and activate a virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

4. Install dependencies with uv:
```bash
uv pip install -e .
```

5. Set up environment variables:
```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key  # Get this from https://serper.dev
QDRANT_URL=your_qdrant_url  # Optional: defaults to http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key  # Optional: if authentication is enabled
```

## Development Setup

Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

Format code:
```bash
black src/
isort src/
```

Run tests:
```bash
pytest
```

## Troubleshooting

### Tiktoken Installation Issues

If you encounter issues with tiktoken installation:

1. Ensure Rust is installed and available in your PATH:
```bash
rustc --version
```

2. If Rust is not found, restart your terminal after installing Rust

3. Try installing tiktoken separately:
```bash
uv pip install --no-deps tiktoken
uv pip install -e .
```

4. On Windows, you might need to install Visual Studio Build Tools with C++ support

## SerperDev Integration

This project uses CrewAI's SerperDevTool for web searches, which provides:
- High-quality search results
- News and recent information
- Structured data extraction
- Multiple search types (web, news, places)

To get your SerperDev API key:
1. Visit https://serper.dev
2. Sign up for an account
3. Navigate to your dashboard
4. Copy your API key
5. Add it to your .env file as SERPER_API_KEY

## Usage

### Python API

Here's a simple example of how to use the Deep Research Agent in your code:

```python
from src.deep_research_agent import DeepResearchAgent

# Initialize the agent
agent = DeepResearchAgent()

# Define your research topic
topic = "Impact of artificial intelligence on healthcare"

# Optional: Provide a structured outline
outline = [
    "Current applications of AI in healthcare",
    "Benefits and improvements in patient care",
    "Challenges and limitations",
    "Future prospects and potential developments",
    "Ethical considerations"
]

# Conduct research
result = agent.research(topic, outline)

# Provide feedback for improvement
feedback = "Please add more specific examples of AI applications in diagnostics"
updated_result = agent.research(topic, outline, feedback)
```

### Local Chat Interface

The project includes a local chat interface with FastAPI backend for interactive debugging and testing. To use it:

1. Start the FastAPI server:
```bash
python src/app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Access the API documentation:
```
http://localhost:5000/docs  # Swagger UI
http://localhost:5000/redoc  # ReDoc
```

The chat interface provides:
- Interactive research topic input
- Optional outline specification
- Real-time research progress updates
- Interactive feedback mechanism
- Easy debugging of the research process
- Process logs panel showing agent's thought process

### API Endpoints

The FastAPI backend provides the following endpoints:

- `GET /` - Serves the chat interface
- `POST /start_research` - Starts a new research session
  ```json
  {
    "topic": "string",
    "outline": ["string"]  // optional
  }
  ```
- `POST /provide_feedback` - Provides feedback for ongoing research
  ```json
  {
    "research_id": "string",
    "feedback": "string"
  }
  ```

Features of the API:
- RESTful endpoints
- Request/response validation with Pydantic
- Interactive API documentation
- CORS support
- WebSocket for real-time updates
- Session management

## How It Works

1. **Initial Research**: The agent uses two powerful tools for comprehensive research:
   - **SerperDev**: For broad web searches and discovering relevant websites
   - **Firecrawl**: For deep analysis of specific websites and content extraction

2. **Structured Analysis**: If an outline is provided, the agent follows it to structure the research. Otherwise, it creates a logical structure based on the initial findings. The agent intelligently combines information from both SerperDev searches and Firecrawl deep analysis.

3. **Human Feedback**: After completing initial research, the agent presents its findings and asks for feedback.

4. **Iterative Improvement**: Based on the feedback received, the agent refines and improves the research until you're satisfied with the results.

5. **Memory Integration**: Research findings and insights are stored in Qdrant for future reference and to improve subsequent research tasks.

## Setting Up Qdrant

1. Using Docker (recommended):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. Or install Qdrant locally following the [official documentation](https://qdrant.tech/documentation/quick_start/).

## Debugging Tips

When using the chat interface for debugging:

1. Monitor the FastAPI server logs for detailed error information
2. Use the interactive API documentation at `/docs` to test endpoints directly
3. Check the browser's developer tools for frontend issues
4. Review the research session management in the interface
5. Monitor Qdrant memory storage and retrieval
6. Use FastAPI's automatic request validation for debugging input issues
7. Check the process logs panel for detailed agent behavior

## Project Management with UV

This project uses uv for modern Python project management:

- Fast dependency resolution
- Deterministic builds
- Built-in virtual environment management
- Compatible with pyproject.toml
- Development tools integration

Key uv commands:
```bash
uv venv                     # Create virtual environment
uv pip install -e .        # Install project dependencies
uv pip install -e ".[dev]" # Install development dependencies
uv pip freeze             # Lock dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.