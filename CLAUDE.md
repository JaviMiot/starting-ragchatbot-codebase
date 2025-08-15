# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (preferred)
chmod +x run.sh
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Dependency Management
**IMPORTANT: Always use `uv` for ALL dependency operations. Never use `pip` directly.**

```bash
# Install all dependencies
uv sync

# Add new dependency
uv add package_name

# Add development dependency
uv add --dev package_name

# Run Python commands with dependencies
uv run python script.py

# Run any command with project dependencies
uv run command_name
```

### Environment Setup
Create `.env` file in root directory:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Architecture Overview

This is a **full-stack RAG (Retrieval-Augmented Generation) chatbot system** for querying course materials using semantic search and AI-powered responses.

### Core Technology Stack
- **Backend**: Python 3.13+, FastAPI, ChromaDB, Anthropic Claude, Sentence Transformers
- **Frontend**: Vanilla HTML/CSS/JavaScript with Marked.js for markdown rendering
- **Package Management**: uv (modern Python package manager)
- **Vector Database**: ChromaDB with dual collections for metadata and content

### Key Architectural Components

#### RAG System (`backend/rag_system.py`)
Central orchestrator that coordinates all system components:
- Document processing and chunking
- Vector storage operations  
- AI generation with tool integration
- Session management for conversations

#### Document Processing (`backend/document_processor.py`)
Handles structured course document format:
```
Course Title: [Course Name]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: [Title]
Lesson Link: [URL]
[Content...]
```
- Parses course metadata and lesson markers
- Smart text chunking (800 chars, 100 overlap)
- Adds contextual prefixes for better retrieval

#### Vector Storage (`backend/vector_store.py`)
Dual ChromaDB collections:
- `course_catalog`: Course metadata for fuzzy name matching
- `course_content`: Actual course content chunks
- Semantic course name resolution
- Advanced filtering by course and lesson

#### Tool-Based Search (`backend/search_tools.py`)
Extensible architecture with `CourseSearchTool` for semantic content retrieval. Tools integrate with Anthropic's tool calling protocol.

#### AI Generation (`backend/ai_generator.py`)
Anthropic Claude integration with:
- Tool-based architecture for dynamic search
- Conversation history integration
- Multi-turn conversations with context

#### Session Management (`backend/session_manager.py`)
Tracks conversation context with configurable history limits (default: 2 exchanges).

### Configuration (`backend/config.py`)
Centralized settings with key parameters:
- Chunk size: 800 characters
- Chunk overlap: 100 characters  
- Max search results: 5
- ChromaDB path: `./chroma_db`

### Data Flow
1. **Document Ingestion**: docs/ folder → Document Processor → Vector Store
2. **Query Processing**: User Query → RAG System → AI Generator → Search Tools → Response
3. **Session Context**: Conversations tracked across interactions

### API Endpoints
- `POST /api/query`: Process user questions with session context
- `GET /api/courses`: Retrieve course statistics
- `GET /`: Frontend interface
- `GET /docs`: FastAPI documentation

### Development Notes
- Documents automatically loaded from `docs/` folder on startup
- Hot reload enabled for development
- Frontend uses no-cache headers for immediate updates
- Type safety enforced throughout with Pydantic models
- CORS configured for frontend-backend communication

### Adding New Course Documents
Place structured course files in `docs/` folder. They will be automatically processed on startup using the expected course format with metadata headers and lesson markers.