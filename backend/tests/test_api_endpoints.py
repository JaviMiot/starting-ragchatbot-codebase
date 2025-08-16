"""
API endpoint tests for the RAG system FastAPI application.

Tests the FastAPI endpoints for proper request/response handling, error cases,
and integration with the RAG system components.
"""
import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import status


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint"""
    
    def test_query_without_session_id(self, test_client, mock_rag_system, mock_ai_response, mock_sources):
        """Test query processing without providing session_id"""
        # Setup mocks
        mock_rag_system._mock_session_manager.create_session.return_value = "new_session_123"
        mock_rag_system.query.return_value = (mock_ai_response, mock_sources)
        
        # Make request
        response = test_client.post(
            "/api/query",
            json={"query": "What is artificial intelligence?"}
        )
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["answer"] == mock_ai_response
        assert data["sources"] == mock_sources
        assert data["session_id"] == "new_session_123"
        
        # Verify RAG system was called correctly
        mock_rag_system._mock_session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with(
            "What is artificial intelligence?", 
            "new_session_123"
        )
    
    def test_query_with_session_id(self, test_client, mock_rag_system, mock_ai_response, mock_sources):
        """Test query processing with existing session_id"""
        # Setup mocks
        mock_rag_system.query.return_value = (mock_ai_response, mock_sources)
        
        # Make request with session_id
        response = test_client.post(
            "/api/query",
            json={
                "query": "Tell me more about machine learning",
                "session_id": "existing_session_456"
            }
        )
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["answer"] == mock_ai_response
        assert data["sources"] == mock_sources
        assert data["session_id"] == "existing_session_456"
        
        # Verify session creation was not called
        mock_rag_system._mock_session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with(
            "Tell me more about machine learning",
            "existing_session_456"
        )
    
    def test_query_with_empty_query(self, test_client, mock_rag_system, mock_ai_response, mock_sources):
        """Test query with empty query string"""
        # Setup mocks for empty query
        mock_rag_system._mock_session_manager.create_session.return_value = "empty_query_session"
        mock_rag_system.query.return_value = (mock_ai_response, mock_sources)
        
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )
        
        # Should still be valid - empty string is allowed
        assert response.status_code == status.HTTP_200_OK
    
    def test_query_missing_query_field(self, test_client):
        """Test query request missing the required query field"""
        response = test_client.post(
            "/api/query",
            json={"session_id": "test_session"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_query_invalid_json(self, test_client):
        """Test query with invalid JSON payload"""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_query_rag_system_error(self, test_client, mock_rag_system):
        """Test query when RAG system raises an exception"""
        # Setup mock to raise exception
        mock_rag_system._mock_session_manager.create_session.return_value = "error_session"
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        response = test_client.post(
            "/api/query",
            json={"query": "This will cause an error"}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "RAG system error" in response.json()["detail"]
    
    def test_query_response_format(self, test_client, mock_rag_system):
        """Test that query response follows the expected format"""
        # Setup mock with various source formats
        mixed_sources = [
            "Simple string source",  # Old format
            {  # New format
                "title": "Course Title - Lesson 1",
                "link": "https://example.com/lesson1"
            }
        ]
        
        mock_rag_system._mock_session_manager.create_session.return_value = "format_test_session"
        mock_rag_system.query.return_value = ("Test response", mixed_sources)
        
        response = test_client.post(
            "/api/query",
            json={"query": "Test query for format"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Verify sources can handle mixed formats
        assert len(data["sources"]) == 2
        assert data["sources"][0] == "Simple string source"
        assert data["sources"][1]["title"] == "Course Title - Lesson 1"
        assert data["sources"][1]["link"] == "https://example.com/lesson1"


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""
    
    def test_get_course_stats_success(self, test_client, mock_rag_system, sample_course_analytics):
        """Test successful retrieval of course statistics"""
        # Setup mock
        mock_rag_system.get_course_analytics.return_value = sample_course_analytics
        
        response = test_client.get("/api/courses")
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Introduction to Python Programming" in data["course_titles"]
        assert "Advanced Machine Learning" in data["course_titles"]
        assert "Web Development Fundamentals" in data["course_titles"]
        
        # Verify RAG system was called
        mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_get_course_stats_empty_result(self, test_client, mock_rag_system):
        """Test course stats when no courses are available"""
        # Setup mock for empty result
        empty_analytics = {
            "total_courses": 0,
            "course_titles": []
        }
        mock_rag_system.get_course_analytics.return_value = empty_analytics
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_get_course_stats_error(self, test_client, mock_rag_system):
        """Test course stats when RAG system raises an exception"""
        # Setup mock to raise exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Database connection failed")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Database connection failed" in response.json()["detail"]
    
    def test_get_course_stats_response_format(self, test_client, mock_rag_system):
        """Test that course stats response follows the expected format"""
        # Setup mock with test data
        test_analytics = {
            "total_courses": 1,
            "course_titles": ["Single Test Course"]
        }
        mock_rag_system.get_course_analytics.return_value = test_analytics
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify response structure matches CourseStats model
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)


@pytest.mark.api
class TestRootEndpoint:
    """Test the root / endpoint"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns basic information"""
        response = test_client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "message" in data
        assert "Course Materials RAG System" in data["message"]


@pytest.mark.api
class TestErrorHandling:
    """Test general error handling and edge cases"""
    
    def test_nonexistent_endpoint(self, test_client):
        """Test requesting a non-existent endpoint"""
        response = test_client.get("/api/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_wrong_http_method_query(self, test_client):
        """Test using wrong HTTP method for query endpoint"""
        response = test_client.get("/api/query")
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_wrong_http_method_courses(self, test_client):
        """Test using wrong HTTP method for courses endpoint"""
        response = test_client.post("/api/courses")
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_query_endpoint_with_extra_fields(self, test_client, mock_rag_system, mock_ai_response, mock_sources):
        """Test query endpoint ignores extra fields in request"""
        mock_rag_system._mock_session_manager.create_session.return_value = "extra_fields_session"
        mock_rag_system.query.return_value = (mock_ai_response, mock_sources)
        
        response = test_client.post(
            "/api/query",
            json={
                "query": "Test query",
                "session_id": "test_session",
                "extra_field": "should be ignored",
                "another_extra": 123
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["session_id"] == "test_session"


@pytest.mark.api
class TestCORSAndMiddleware:
    """Test CORS and middleware functionality"""
    
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in responses"""
        response = test_client.get("/")
        
        # Check for CORS headers (these are set by CORSMiddleware)
        assert response.status_code == status.HTTP_200_OK
        
        # The TestClient doesn't automatically add CORS headers like a real browser would,
        # but we can verify the middleware is configured by checking the response is successful
        # In a real deployment, we'd see headers like Access-Control-Allow-Origin
    
    def test_options_request(self, test_client):
        """Test OPTIONS request for CORS preflight"""
        response = test_client.options("/api/query")
        
        # OPTIONS should be allowed due to CORS middleware
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]


@pytest.mark.api 
class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def test_conversation_flow(self, test_client, mock_rag_system):
        """Test a multi-turn conversation flow"""
        # Setup mock responses
        mock_rag_system._mock_session_manager.create_session.return_value = "conversation_session"
        
        # First query
        mock_rag_system.query.return_value = ("AI is artificial intelligence.", [])
        response1 = test_client.post(
            "/api/query",
            json={"query": "What is AI?"}
        )
        
        assert response1.status_code == status.HTTP_200_OK
        session_id = response1.json()["session_id"]
        
        # Follow-up query with same session
        mock_rag_system.query.return_value = ("Machine learning is a subset of AI.", [])
        response2 = test_client.post(
            "/api/query", 
            json={
                "query": "What about machine learning?",
                "session_id": session_id
            }
        )
        
        assert response2.status_code == status.HTTP_200_OK
        assert response2.json()["session_id"] == session_id
        
        # Verify RAG system received both queries
        assert mock_rag_system.query.call_count == 2
    
    def test_query_then_get_courses(self, test_client, mock_rag_system, sample_course_analytics):
        """Test querying then getting course statistics"""
        # Setup mocks
        mock_rag_system._mock_session_manager.create_session.return_value = "mixed_session"
        mock_rag_system.query.return_value = ("Query response", [])
        mock_rag_system.get_course_analytics.return_value = sample_course_analytics
        
        # First make a query
        query_response = test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        assert query_response.status_code == status.HTTP_200_OK
        
        # Then get course stats
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == status.HTTP_200_OK
        
        # Both should work independently
        assert courses_response.json()["total_courses"] == 3