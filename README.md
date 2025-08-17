# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.

### Key Features

- **Semantic Search**: Vector-based search through course materials using ChromaDB
- **AI-Powered Responses**: Context-aware answers using Anthropic's Claude AI
- **Tool Integration**: Extensible search tools for dynamic content retrieval
- **Session Management**: Conversation history tracking across interactions
- **Dual Vector Storage**: Separate collections for course metadata and content
- **Smart Document Processing**: Structured course document parsing with metadata

## Architecture

### Technology Stack
- **Backend**: Python 3.13+, FastAPI, ChromaDB, Anthropic Claude, Sentence Transformers
- **Frontend**: Vanilla HTML/CSS/JavaScript with Marked.js for markdown rendering
- **Package Management**: uv (modern Python package manager)
- **Vector Database**: ChromaDB with dual collections

### Core Components
- **RAG System** (`backend/rag_system.py`): Central orchestrator
- **Document Processor** (`backend/document_processor.py`): Handles structured course documents
- **Vector Store** (`backend/vector_store.py`): Dual ChromaDB collections management
- **Search Tools** (`backend/search_tools.py`): Tool-based semantic search
- **AI Generator** (`backend/ai_generator.py`): Anthropic Claude integration
- **Session Manager** (`backend/session_manager.py`): Conversation context tracking

## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key (for Claude AI)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- **Web Interface**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`

## API Endpoints

- `POST /api/query` - Process user questions with session context
- `GET /api/courses` - Retrieve course statistics
- `GET /` - Frontend interface
- `GET /docs` - FastAPI interactive documentation

## Adding Course Documents

Place structured course files in the `docs/` folder. The system expects this format:

```
Course Title: [Course Name]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: [Title]
Lesson Link: [URL]
[Content...]

Lesson 1: [Title]
Lesson Link: [URL]
[Content...]
```

Documents are automatically processed on startup with:
- Smart text chunking (800 characters, 100 overlap)
- Metadata extraction from headers
- Contextual prefixes for better retrieval

## Development

### Package Management

**IMPORTANT**: Always use `uv` for dependency operations:

```bash
# Install dependencies
uv sync

# Add new dependency
uv add package_name

# Add development dependency
uv add --dev package_name

# Run commands with project dependencies
uv run python script.py
uv run command_name
```

### Testing

Run the test suite:
```bash
uv run pytest
```

Test configuration is in `pyproject.toml` with markers for:
- `unit`: Individual component tests
- `integration`: Component interaction tests
- `api`: API endpoint tests

### Configuration

Key settings in `backend/config.py`:
- Chunk size: 800 characters
- Chunk overlap: 100 characters
- Max search results: 5
- ChromaDB path: `./chroma_db`
- Session history limit: 2 exchanges

## Usage Examples

### Query Interface

The web interface provides a chat-like experience where you can:
- Ask questions about course materials
- Get context-aware responses with source references
- Maintain conversation history across sessions

### Example Queries

```
"What is the main topic of Course 1?"
"Explain the concept covered in Lesson 3 of the Python course"
"Compare the approaches mentioned in different courses"
```

## Project Structure

```
├── backend/
│   ├── app.py              # FastAPI application
│   ├── rag_system.py       # Main RAG orchestrator
│   ├── document_processor.py # Course document parsing
│   ├── vector_store.py     # ChromaDB operations
│   ├── search_tools.py     # Semantic search tools
│   ├── ai_generator.py     # Claude AI integration
│   ├── session_manager.py  # Conversation tracking
│   ├── config.py           # Configuration settings
│   ├── models.py           # Pydantic data models
│   └── tests/              # Test suite
├── frontend/
│   ├── index.html          # Web interface
│   ├── script.js           # Frontend logic
│   └── style.css           # Styling
├── docs/                   # Course material files
├── run.sh                  # Quick start script
├── pyproject.toml          # Project dependencies
└── CLAUDE.md               # Development guidelines
```

## Contributing

1. Follow the development guidelines in `CLAUDE.md`
2. Use `uv` for all dependency management
3. Run tests before submitting changes: `uv run pytest`
4. Ensure type safety with Pydantic models
5. Follow the existing code conventions and patterns

## License

This project is licensed under the MIT License.

