"""
Shared fixtures and configuration for testing the RAG system.
"""
import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from typing import Generator, Dict, Any

# Add backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config


class MockConfig:
    """Mock configuration for testing"""
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CHROMA_PATH = "./test_chroma_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_RESULTS = 5
    ANTHROPIC_API_KEY = "test_api_key"
    ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    MAX_HISTORY = 2


@pytest.fixture
def mock_config():
    """Provide mock configuration for tests"""
    return MockConfig()


@pytest.fixture
def mock_rag_system():
    """Create a fully mocked RAG system for testing"""
    config = MockConfig()
    
    # Create mocks for all components
    mock_document_processor = Mock()
    mock_vector_store = Mock()
    mock_ai_generator = Mock()
    mock_session_manager = Mock()
    
    with patch('rag_system.DocumentProcessor') as mock_doc_proc:
        with patch('rag_system.VectorStore') as mock_vector:
            with patch('rag_system.AIGenerator') as mock_ai:
                with patch('rag_system.SessionManager') as mock_session:
                    mock_doc_proc.return_value = mock_document_processor
                    mock_vector.return_value = mock_vector_store
                    mock_ai.return_value = mock_ai_generator
                    mock_session.return_value = mock_session_manager
                    
                    rag_system = RAGSystem(config)
                    
    # Attach mocks to the rag_system for easy access in tests
    rag_system._mock_document_processor = mock_document_processor
    rag_system._mock_vector_store = mock_vector_store
    rag_system._mock_ai_generator = mock_ai_generator
    rag_system._mock_session_manager = mock_session_manager
    
    # Mock the actual methods on the rag_system instance
    rag_system.query = Mock()
    rag_system.get_course_analytics = Mock()
    rag_system.add_course_document = Mock()
    rag_system.add_course_folder = Mock()
    
    return rag_system


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mocked dependencies"""
    from pydantic import BaseModel
    from typing import List, Optional, Union, Dict, Any
    from fastapi import HTTPException
    
    # Create test app without static file mounting to avoid missing frontend files
    app = FastAPI(title="Test Course Materials RAG System")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models (copied from main app)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Dict[str, Any]]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Define API endpoints inline to avoid import issues with static files
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            # Create session if not provided
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            # Process query using RAG system
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        """Root endpoint for testing"""
        return {"message": "Course Materials RAG System"}
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)


@pytest.fixture
def sample_course():
    """Provide a sample course for testing"""
    return Course(
        title="Test Course: Introduction to AI",
        course_link="https://example.com/ai-course",
        instructor="Dr. Test Instructor"
    )


@pytest.fixture
def sample_chunks():
    """Provide sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is the first chunk about AI fundamentals.",
            course_title="Test Course: Introduction to AI",
            lesson_title="Lesson 1: What is AI?",
            lesson_number=1,
            lesson_link="https://example.com/lesson1",
            chunk_index=0
        ),
        CourseChunk(
            content="This is the second chunk about machine learning basics.",
            course_title="Test Course: Introduction to AI", 
            lesson_title="Lesson 2: Machine Learning",
            lesson_number=2,
            lesson_link="https://example.com/lesson2",
            chunk_index=1
        )
    ]


@pytest.fixture
def sample_search_results():
    """Provide sample search results for testing"""
    return SearchResults(
        documents=[
            "AI is a branch of computer science that focuses on creating intelligent machines.",
            "Machine learning is a subset of AI that enables computers to learn without explicit programming."
        ],
        metadata=[
            {
                "course_title": "Test Course: Introduction to AI",
                "lesson_title": "Lesson 1: What is AI?",
                "lesson_number": 1,
                "chunk_index": 0
            },
            {
                "course_title": "Test Course: Introduction to AI", 
                "lesson_title": "Lesson 2: Machine Learning",
                "lesson_number": 2,
                "chunk_index": 1
            }
        ],
        distances=[0.1, 0.2],
        error=None
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_course_analytics():
    """Provide sample course analytics data"""
    return {
        "total_courses": 3,
        "course_titles": [
            "Introduction to Python Programming",
            "Advanced Machine Learning", 
            "Web Development Fundamentals"
        ]
    }


# Mock responses for different test scenarios
@pytest.fixture
def mock_ai_response():
    """Sample AI response for testing"""
    return "Based on the course materials, AI (Artificial Intelligence) is a branch of computer science that focuses on creating intelligent machines capable of performing tasks that typically require human intelligence."


@pytest.fixture
def mock_sources():
    """Sample sources data for testing"""
    return [
        {
            "title": "Test Course: Introduction to AI - Lesson 1: What is AI?",
            "link": "https://example.com/lesson1"
        },
        {
            "title": "Test Course: Introduction to AI - Lesson 2: Machine Learning", 
            "link": "https://example.com/lesson2"
        }
    ]


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Auto-used fixture to set up test environment"""
    # Suppress warnings that might appear during testing
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")
    
    # Set environment variables for testing
    os.environ["ANTHROPIC_API_KEY"] = "test_api_key_for_testing"
    
    yield
    
    # Cleanup after tests
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]