import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Add backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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


class TestRAGSystem:
    """Test RAG System integration and end-to-end functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = MockConfig()
        
        # Create mocks for all components
        self.mock_document_processor = Mock()
        self.mock_vector_store = Mock()
        self.mock_ai_generator = Mock()
        self.mock_session_manager = Mock()
        
        # Patch all component classes
        with patch('rag_system.DocumentProcessor') as mock_doc_proc:
            with patch('rag_system.VectorStore') as mock_vector:
                with patch('rag_system.AIGenerator') as mock_ai:
                    with patch('rag_system.SessionManager') as mock_session:
                        mock_doc_proc.return_value = self.mock_document_processor
                        mock_vector.return_value = self.mock_vector_store
                        mock_ai.return_value = self.mock_ai_generator
                        mock_session.return_value = self.mock_session_manager
                        
                        self.rag_system = RAGSystem(self.config)
    
    def test_init(self):
        """Test RAG System initialization"""
        assert self.rag_system.config == self.config
        assert self.rag_system.document_processor == self.mock_document_processor
        assert self.rag_system.vector_store == self.mock_vector_store
        assert self.rag_system.ai_generator == self.mock_ai_generator
        assert self.rag_system.session_manager == self.mock_session_manager
        
        # Verify tool manager was setup
        assert self.rag_system.tool_manager is not None
        assert self.rag_system.search_tool is not None
        assert self.rag_system.outline_tool is not None
    
    def test_add_course_document_success(self):
        """Test successful course document addition"""
        # Setup mock course and chunks
        test_course = Course(
            title="Test Course",
            course_link="http://test.com",
            instructor="Test Instructor"
        )
        test_chunks = [
            CourseChunk(content="Chunk 1", course_title="Test Course", chunk_index=0),
            CourseChunk(content="Chunk 2", course_title="Test Course", chunk_index=1)
        ]
        
        self.mock_document_processor.process_course_document.return_value = (test_course, test_chunks)
        
        # Execute
        course, chunk_count = self.rag_system.add_course_document("/path/to/course.txt")
        
        # Assertions
        assert course == test_course
        assert chunk_count == 2
        
        # Verify calls
        self.mock_document_processor.process_course_document.assert_called_once_with("/path/to/course.txt")
        self.mock_vector_store.add_course_metadata.assert_called_once_with(test_course)
        self.mock_vector_store.add_course_content.assert_called_once_with(test_chunks)
    
    def test_add_course_document_failure(self):
        """Test course document addition with processing error"""
        self.mock_document_processor.process_course_document.side_effect = Exception("Processing error")
        
        course, chunk_count = self.rag_system.add_course_document("/path/to/course.txt")
        
        assert course is None
        assert chunk_count == 0
        
        # Verify vector store was not called
        self.mock_vector_store.add_course_metadata.assert_not_called()
        self.mock_vector_store.add_course_content.assert_not_called()
    
    def test_query_without_session(self):
        """Test query processing without session context"""
        # Setup mock AI response
        expected_response = "This is the AI response"
        expected_sources = [{"title": "Test Course - Lesson 1", "link": "http://lesson1.com"}]
        
        self.mock_ai_generator.generate_response.return_value = expected_response
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=expected_sources)
        
        # Execute query
        response, sources = self.rag_system.query("What is AI?")
        
        # Assertions
        assert response == expected_response
        assert sources == expected_sources
        
        # Verify AI generator was called correctly
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        
        assert "What is AI?" in call_args["query"]
        assert call_args["conversation_history"] is None
        assert call_args["tools"] is not None
        assert call_args["tool_manager"] == self.rag_system.tool_manager
        
        # Verify session manager was not used
        self.mock_session_manager.get_conversation_history.assert_not_called()
        self.mock_session_manager.add_exchange.assert_not_called()
    
    def test_query_with_session(self):
        """Test query processing with session context"""
        session_id = "test_session_123"
        conversation_history = "User: Previous question\nAssistant: Previous answer"
        expected_response = "This is the AI response with context"
        
        self.mock_session_manager.get_conversation_history.return_value = conversation_history
        self.mock_ai_generator.generate_response.return_value = expected_response
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        
        # Execute query with session
        response, sources = self.rag_system.query("Follow up question", session_id)
        
        # Assertions
        assert response == expected_response
        assert sources == []
        
        # Verify session history was retrieved
        self.mock_session_manager.get_conversation_history.assert_called_once_with(session_id)
        
        # Verify AI generator got conversation history
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        assert call_args["conversation_history"] == conversation_history
        
        # Verify conversation was updated
        self.mock_session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow up question", expected_response
        )
        
        # Verify sources were reset
        self.rag_system.tool_manager.reset_sources.assert_called_once()
    
    def test_query_prompt_format(self):
        """Test that query prompt is formatted correctly"""
        self.mock_ai_generator.generate_response.return_value = "Response"
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        
        self.rag_system.query("What is machine learning?")
        
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        prompt = call_args["query"]
        
        assert "Answer this question about course materials:" in prompt
        assert "What is machine learning?" in prompt
    
    def test_add_course_folder_success(self):
        """Test successful course folder processing"""
        # Setup mock directory structure
        test_files = ["course1.txt", "course2.pdf", "course3.docx", "readme.md"]
        
        # Mock course objects
        course1 = Course(title="Course 1")
        course2 = Course(title="Course 2")
        course3 = Course(title="Course 3")
        
        chunks1 = [CourseChunk(content="C1 chunk", course_title="Course 1", chunk_index=0)]
        chunks2 = [CourseChunk(content="C2 chunk", course_title="Course 2", chunk_index=0)]
        chunks3 = [CourseChunk(content="C3 chunk", course_title="Course 3", chunk_index=0)]
        
        self.mock_document_processor.process_course_document.side_effect = [
            (course1, chunks1),
            (course2, chunks2), 
            (course3, chunks3)
        ]
        
        # Mock existing course titles (empty)
        self.mock_vector_store.get_existing_course_titles.return_value = []
        
        with patch('os.path.exists') as mock_exists:
            with patch('os.listdir') as mock_listdir:
                with patch('os.path.isfile') as mock_isfile:
                    mock_exists.return_value = True
                    mock_listdir.return_value = test_files
                    mock_isfile.side_effect = lambda path: path.endswith(('.txt', '.pdf', '.docx'))
                    
                    courses_added, chunks_added = self.rag_system.add_course_folder("/fake/docs")
        
        # Assertions
        assert courses_added == 3
        assert chunks_added == 3
        
        # Verify all courses were processed
        assert self.mock_document_processor.process_course_document.call_count == 3
        assert self.mock_vector_store.add_course_metadata.call_count == 3
        assert self.mock_vector_store.add_course_content.call_count == 3
    
    def test_add_course_folder_skip_existing(self):
        """Test that existing courses are skipped"""
        test_files = ["course1.txt", "course2.txt"]
        
        course1 = Course(title="Course 1")
        course2 = Course(title="Course 2")
        chunks1 = [CourseChunk(content="C1 chunk", course_title="Course 1", chunk_index=0)]
        chunks2 = [CourseChunk(content="C2 chunk", course_title="Course 2", chunk_index=0)]
        
        self.mock_document_processor.process_course_document.side_effect = [
            (course1, chunks1),
            (course2, chunks2)
        ]
        
        # Mock existing courses (Course 1 already exists)
        self.mock_vector_store.get_existing_course_titles.return_value = ["Course 1"]
        
        with patch('os.path.exists') as mock_exists:
            with patch('os.listdir') as mock_listdir:
                with patch('os.path.isfile') as mock_isfile:
                    mock_exists.return_value = True
                    mock_listdir.return_value = test_files
                    mock_isfile.return_value = True
                    
                    courses_added, chunks_added = self.rag_system.add_course_folder("/fake/docs")
        
        # Assertions - only Course 2 should be added
        assert courses_added == 1
        assert chunks_added == 1
        
        # Verify Course 1 was processed but not added
        assert self.mock_document_processor.process_course_document.call_count == 2
        assert self.mock_vector_store.add_course_metadata.call_count == 1
        assert self.mock_vector_store.add_course_content.call_count == 1
    
    def test_add_course_folder_clear_existing(self):
        """Test folder processing with clear_existing=True"""
        with patch('os.path.exists') as mock_exists:
            with patch('os.listdir') as mock_listdir:
                mock_exists.return_value = True
                mock_listdir.return_value = []
                
                self.rag_system.add_course_folder("/fake/docs", clear_existing=True)
        
        # Verify clear was called
        self.mock_vector_store.clear_all_data.assert_called_once()
    
    def test_add_course_folder_nonexistent_path(self):
        """Test folder processing with non-existent path"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            courses_added, chunks_added = self.rag_system.add_course_folder("/nonexistent/path")
        
        assert courses_added == 0
        assert chunks_added == 0
    
    def test_get_course_analytics(self):
        """Test getting course analytics"""
        expected_analytics = {
            "total_courses": 5,
            "course_titles": ["Course A", "Course B", "Course C", "Course D", "Course E"]
        }
        
        self.mock_vector_store.get_course_count.return_value = 5
        self.mock_vector_store.get_existing_course_titles.return_value = [
            "Course A", "Course B", "Course C", "Course D", "Course E"
        ]
        
        analytics = self.rag_system.get_course_analytics()
        
        assert analytics == expected_analytics
        self.mock_vector_store.get_course_count.assert_called_once()
        self.mock_vector_store.get_existing_course_titles.assert_called_once()


class TestRAGSystemWithToolIntegration:
    """Test RAG System with actual tool integration"""
    
    def setup_method(self):
        """Setup test with partial mocking for tool integration"""
        self.config = MockConfig()
        
        # Mock external dependencies but keep tool system intact
        self.mock_document_processor = Mock()
        self.mock_vector_store = Mock()
        self.mock_ai_generator = Mock()
        self.mock_session_manager = Mock()
        
        with patch('rag_system.DocumentProcessor') as mock_doc_proc:
            with patch('rag_system.VectorStore') as mock_vector:
                with patch('rag_system.AIGenerator') as mock_ai:
                    with patch('rag_system.SessionManager') as mock_session:
                        mock_doc_proc.return_value = self.mock_document_processor
                        mock_vector.return_value = self.mock_vector_store
                        mock_ai.return_value = self.mock_ai_generator
                        mock_session.return_value = self.mock_session_manager
                        
                        self.rag_system = RAGSystem(self.config)
    
    def test_tool_manager_setup(self):
        """Test that tool manager is properly setup with tools"""
        # Verify tool manager has both tools registered
        tool_definitions = self.rag_system.tool_manager.get_tool_definitions()
        
        assert len(tool_definitions) == 2
        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
    
    def test_search_tool_integration(self):
        """Test search tool integration with RAG system"""
        # Setup mock search results
        mock_results = SearchResults(
            documents=["Test course content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "http://lesson1.com"
        
        # Execute search through tool manager
        result = self.rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="test query",
            course_name="Test Course"
        )
        
        assert "Test Course" in result
        assert "Test course content" in result
        
        # Verify vector store was called
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course", 
            lesson_number=None
        )
        
        # Verify sources were tracked
        sources = self.rag_system.tool_manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["title"] == "Test Course - Lesson 1"
        assert sources[0]["link"] == "http://lesson1.com"
    
    def test_outline_tool_integration(self):
        """Test outline tool integration with RAG system"""
        # Setup mock course metadata
        lessons_json = '[{"lesson_number": 1, "lesson_title": "Introduction"}, {"lesson_number": 2, "lesson_title": "Advanced"}]'
        
        self.mock_vector_store._resolve_course_name.return_value = "Full Course Title"
        self.mock_vector_store.course_catalog.get.return_value = {
            'metadatas': [{
                'title': 'Full Course Title',
                'course_link': 'http://course.com',
                'instructor': 'Test Instructor',
                'lessons_json': lessons_json
            }]
        }
        
        # Execute outline tool
        result = self.rag_system.tool_manager.execute_tool(
            "get_course_outline",
            course_name="Course"
        )
        
        assert "Full Course Title" in result
        assert "Test Instructor" in result
        assert "http://course.com" in result
        assert "1. Introduction" in result
        assert "2. Advanced" in result
    
    def test_end_to_end_query_flow(self):
        """Test complete end-to-end query flow"""
        # Setup AI generator to simulate tool use
        def mock_generate_response(*args, **kwargs):
            tool_manager = kwargs.get('tool_manager')
            if tool_manager:
                # Simulate AI using the search tool
                tool_result = tool_manager.execute_tool(
                    "search_course_content",
                    query="AI fundamentals",
                    course_name="AI Course"
                )
                return f"Based on the course materials: {tool_result[:50]}..."
            return "Direct response without tools"
        
        self.mock_ai_generator.generate_response.side_effect = mock_generate_response
        
        # Setup search results
        mock_results = SearchResults(
            documents=["AI is a branch of computer science that focuses on creating intelligent machines."],
            metadata=[{"course_title": "AI Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "http://ai-lesson1.com"
        
        # Execute query
        response, sources = self.rag_system.query("What is AI?")
        
        # Verify response includes search results
        assert "Based on the course materials:" in response
        assert "AI is a branch of computer science" in response
        
        # Verify sources were captured
        assert len(sources) == 1
        assert sources[0]["title"] == "AI Course - Lesson 1"
        assert sources[0]["link"] == "http://ai-lesson1.com"
        
        # Verify AI generator was called with tools
        self.mock_ai_generator.generate_response.assert_called_once()
        call_kwargs = self.mock_ai_generator.generate_response.call_args[1]
        assert call_kwargs["tools"] is not None
        assert call_kwargs["tool_manager"] == self.rag_system.tool_manager


if __name__ == "__main__":
    pytest.main([__file__, "-v"])