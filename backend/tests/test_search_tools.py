import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Add backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test the CourseSearchTool execute method and functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create mock vector store
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_execute_successful_search(self):
        """Test execute method with successful search results"""
        # Setup mock search results
        mock_results = SearchResults(
            documents=["Course content chunk 1", "Course content chunk 2"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        # Execute search
        result = self.search_tool.execute("test query", "Test Course")
        
        # Assertions
        assert isinstance(result, str)
        assert "Test Course" in result
        assert "Lesson 1" in result
        assert "Lesson 2" in result
        assert "Course content chunk 1" in result
        assert "Course content chunk 2" in result
        
        # Verify vector store was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course",
            lesson_number=None
        )
    
    def test_execute_with_lesson_filter(self):
        """Test execute method with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query", "Test Course", 3)
        
        assert "Lesson 3" in result
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course",
            lesson_number=3
        )
    
    def test_execute_with_search_error(self):
        """Test execute method when vector store returns error"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="ChromaDB connection failed"
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        assert result == "ChromaDB connection failed"
    
    def test_execute_empty_results(self):
        """Test execute method with no search results"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent query", "Nonexistent Course")
        
        expected = "No relevant content found in course 'Nonexistent Course'."
        assert result == expected
    
    def test_execute_empty_results_with_lesson(self):
        """Test execute method with no results when filtering by lesson"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query", "Test Course", 99)
        
        expected = "No relevant content found in course 'Test Course' in lesson 99."
        assert result == expected
    
    def test_execute_no_filters(self):
        """Test execute method without course or lesson filters"""
        mock_results = SearchResults(
            documents=["General content"],
            metadata=[{"course_title": "Various Course", "lesson_number": None}],
            distances=[0.1],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("general query")
        
        assert "Various Course" in result
        assert "General content" in result
    
    def test_format_results_with_lesson_links(self):
        """Test result formatting includes lesson links when available"""
        mock_results = SearchResults(
            documents=["Content with link"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        result = self.search_tool.execute("test query")
        
        # Check that sources were tracked with links
        assert len(self.search_tool.last_sources) == 1
        source = self.search_tool.last_sources[0]
        assert source["title"] == "Test Course - Lesson 1"
        assert source["link"] == "https://example.com/lesson1"
    
    def test_get_tool_definition(self):
        """Test that tool definition is properly formatted"""
        definition = self.search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        
        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties


class TestToolManager:
    """Test the ToolManager functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.tool_manager = ToolManager()
        self.mock_vector_store = Mock()
    
    def test_register_tool(self):
        """Test tool registration"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        self.tool_manager.register_tool(search_tool)
        
        assert "search_course_content" in self.tool_manager.tools
        
        definitions = self.tool_manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
    
    def test_execute_tool(self):
        """Test tool execution through manager"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        
        # Mock successful search
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        self.tool_manager.register_tool(search_tool)
        
        result = self.tool_manager.execute_tool(
            "search_course_content", 
            query="test query",
            course_name="Test Course"
        )
        
        assert isinstance(result, str)
        assert "Test Course" in result
    
    def test_execute_nonexistent_tool(self):
        """Test execution of non-existent tool"""
        result = self.tool_manager.execute_tool("nonexistent_tool", query="test")
        
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_get_last_sources(self):
        """Test retrieval of last search sources"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        
        # Setup mock with sources
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        self.tool_manager.register_tool(search_tool)
        
        # Execute search to populate sources
        self.tool_manager.execute_tool("search_course_content", query="test")
        
        sources = self.tool_manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["title"] == "Test Course - Lesson 1"
    
    def test_reset_sources(self):
        """Test resetting of sources"""
        search_tool = CourseSearchTool(self.mock_vector_store)
        self.tool_manager.register_tool(search_tool)
        
        # Add some sources
        search_tool.last_sources = [{"title": "Test", "link": "test.com"}]
        
        self.tool_manager.reset_sources()
        
        assert search_tool.last_sources == []


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])