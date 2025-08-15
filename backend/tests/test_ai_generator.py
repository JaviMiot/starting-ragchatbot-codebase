import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Add backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class MockAnthropicResponse:
    """Mock Anthropic API response"""
    def __init__(self, content, stop_reason="end_turn", tool_use_content=None):
        self.content = content if isinstance(content, list) else [MockContentBlock(content)]
        self.stop_reason = stop_reason
        if tool_use_content:
            self.content = tool_use_content


class MockContentBlock:
    """Mock content block from Anthropic API"""
    def __init__(self, text=None, tool_use_data=None):
        self.type = "text" if text else "tool_use"
        self.text = text
        if tool_use_data:
            self.type = "tool_use"
            self.name = tool_use_data.get("name")
            self.id = tool_use_data.get("id", "tool_123")
            self.input = tool_use_data.get("input", {})


class TestAIGenerator:
    """Test AI Generator functionality and tool integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.api_key = "test_api_key"
        self.model = "claude-sonnet-4-20250514"
        
        # Mock the Anthropic client
        self.mock_client = Mock()
        
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator(self.api_key, self.model)
    
    def test_init(self):
        """Test AI Generator initialization"""
        assert self.ai_generator.model == self.model
        assert self.ai_generator.base_params["model"] == self.model
        assert self.ai_generator.base_params["temperature"] == 0
        assert self.ai_generator.base_params["max_tokens"] == 800
    
    def test_generate_response_without_tools(self):
        """Test basic response generation without tools"""
        # Setup mock response
        mock_response = MockAnthropicResponse("This is a test response")
        self.mock_client.messages.create.return_value = mock_response
        
        result = self.ai_generator.generate_response("What is AI?")
        
        assert result == "This is a test response"
        
        # Verify API was called correctly
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args[1]
        
        assert call_args["model"] == self.model
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "What is AI?"
        assert "tools" not in call_args
    
    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history"""
        mock_response = MockAnthropicResponse("Response with context")
        self.mock_client.messages.create.return_value = mock_response
        
        history = "User: Previous question\nAssistant: Previous answer"
        
        result = self.ai_generator.generate_response(
            "Follow up question", 
            conversation_history=history
        )
        
        assert result == "Response with context"
        
        # Verify system prompt includes history
        call_args = self.mock_client.messages.create.call_args[1]
        assert history in call_args["system"]
    
    def test_generate_response_with_tools_no_tool_use(self):
        """Test response generation with tools available but no tool use"""
        mock_response = MockAnthropicResponse("Direct answer without tools")
        self.mock_client.messages.create.return_value = mock_response
        
        tools = [{"name": "search_course_content", "description": "Search courses"}]
        
        result = self.ai_generator.generate_response(
            "General question",
            tools=tools
        )
        
        assert result == "Direct answer without tools"
        
        # Verify tools were passed to API
        call_args = self.mock_client.messages.create.call_args[1]
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_tool_use(self):
        """Test response generation that uses tools"""
        # Setup tool use response
        tool_use_content = [
            MockContentBlock(tool_use_data={
                "name": "search_course_content",
                "id": "tool_123",
                "input": {"query": "test query", "course_name": "Test Course"}
            })
        ]
        
        initial_response = MockAnthropicResponse(
            content=None,
            stop_reason="tool_use",
            tool_use_content=tool_use_content
        )
        
        final_response = MockAnthropicResponse("Here's the answer based on search results")
        
        # Setup mock to return different responses for consecutive calls
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Setup mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result content"
        
        tools = [{"name": "search_course_content", "description": "Search courses"}]
        
        result = self.ai_generator.generate_response(
            "Question about course content",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Here's the answer based on search results"
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query",
            course_name="Test Course"
        )
        
        # Verify two API calls were made
        assert self.mock_client.messages.create.call_count == 2
    
    def test_handle_tool_execution_single_tool(self):
        """Test tool execution handling with single tool call"""
        # Create mock initial response with tool use
        tool_content = MockContentBlock(tool_use_data={
            "name": "search_course_content",
            "id": "tool_123",
            "input": {"query": "AI basics"}
        })
        
        initial_response = MockAnthropicResponse(
            content=None,
            tool_use_content=[tool_content]
        )
        
        # Mock final response after tool execution
        final_response = MockAnthropicResponse("Based on the search, here's the answer")
        self.mock_client.messages.create.return_value = final_response
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Course content about AI"
        
        # Mock base params
        base_params = {
            "messages": [{"role": "user", "content": "What is AI?"}],
            "system": "Test system prompt",
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
        
        result = self.ai_generator._handle_tool_execution(
            initial_response, 
            base_params, 
            mock_tool_manager
        )
        
        assert result == "Based on the search, here's the answer"
        
        # Verify tool execution
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="AI basics"
        )
        
        # Verify final API call structure
        final_call_args = self.mock_client.messages.create.call_args[1]
        assert len(final_call_args["messages"]) == 3  # Original + assistant + tool_result
        assert final_call_args["messages"][1]["role"] == "assistant"
        assert final_call_args["messages"][2]["role"] == "user"
        
        # Check tool result structure
        tool_result_content = final_call_args["messages"][2]["content"]
        assert len(tool_result_content) == 1
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "tool_123"
        assert tool_result_content[0]["content"] == "Course content about AI"
    
    def test_handle_tool_execution_multiple_tools(self):
        """Test tool execution with multiple tool calls"""
        # Create mock response with multiple tool uses
        tool_content_1 = MockContentBlock(tool_use_data={
            "name": "search_course_content",
            "id": "tool_123",
            "input": {"query": "AI basics", "course_name": "Course 1"}
        })
        
        tool_content_2 = MockContentBlock(tool_use_data={
            "name": "search_course_content", 
            "id": "tool_456",
            "input": {"query": "ML fundamentals", "course_name": "Course 2"}
        })
        
        initial_response = MockAnthropicResponse(
            content=None,
            tool_use_content=[tool_content_1, tool_content_2]
        )
        
        final_response = MockAnthropicResponse("Combined answer from multiple searches")
        self.mock_client.messages.create.return_value = final_response
        
        # Mock tool manager with different returns for each call
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "AI content from Course 1",
            "ML content from Course 2"
        ]
        
        base_params = {
            "messages": [{"role": "user", "content": "Compare AI and ML"}],
            "system": "Test system prompt",
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
        
        result = self.ai_generator._handle_tool_execution(
            initial_response,
            base_params,
            mock_tool_manager
        )
        
        assert result == "Combined answer from multiple searches"
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify final message structure includes both tool results
        final_call_args = self.mock_client.messages.create.call_args[1]
        tool_result_content = final_call_args["messages"][2]["content"]
        assert len(tool_result_content) == 2
        
        # Check both tool results
        assert tool_result_content[0]["tool_use_id"] == "tool_123"
        assert tool_result_content[0]["content"] == "AI content from Course 1"
        assert tool_result_content[1]["tool_use_id"] == "tool_456"
        assert tool_result_content[1]["content"] == "ML content from Course 2"
    
    def test_api_error_handling(self):
        """Test handling of API errors"""
        # Mock API exception
        self.mock_client.messages.create.side_effect = Exception("API Error: Invalid API key")
        
        with pytest.raises(Exception) as exc_info:
            self.ai_generator.generate_response("Test question")
        
        assert "API Error: Invalid API key" in str(exc_info.value)
    
    def test_system_prompt_format(self):
        """Test that system prompt is properly formatted"""
        mock_response = MockAnthropicResponse("Test response")
        self.mock_client.messages.create.return_value = mock_response
        
        self.ai_generator.generate_response("Test question")
        
        call_args = self.mock_client.messages.create.call_args[1]
        system_prompt = call_args["system"]
        
        # Verify system prompt contains key instructions
        assert "You are an AI assistant specialized in course materials" in system_prompt
        assert "Course outline queries" in system_prompt
        assert "Content search queries" in system_prompt
        assert "search_course_content tool" in system_prompt


class TestAIGeneratorIntegration:
    """Integration tests with real tool manager"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        self.api_key = "test_api_key"
        self.model = "claude-sonnet-4-20250514"
        
        # Create real tool manager with mocked vector store
        self.mock_vector_store = Mock()
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
        self.tool_manager.register_tool(self.search_tool)
        
        # Mock Anthropic client
        self.mock_client = Mock()
        
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator(self.api_key, self.model)
    
    def test_full_integration_with_tool_manager(self):
        """Test full integration between AI generator and tool manager"""
        from vector_store import SearchResults
        
        # Setup mock search results
        mock_results = SearchResults(
            documents=["This is course content about AI fundamentals"],
            metadata=[{"course_title": "AI Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        # Setup tool use response
        tool_content = MockContentBlock(tool_use_data={
            "name": "search_course_content",
            "id": "tool_123", 
            "input": {"query": "AI fundamentals", "course_name": "AI Course"}
        })
        
        initial_response = MockAnthropicResponse(
            content=None,
            stop_reason="tool_use",
            tool_use_content=[tool_content]
        )
        
        final_response = MockAnthropicResponse("AI is a field of computer science...")
        
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Execute with real tool manager
        result = self.ai_generator.generate_response(
            "What is AI?",
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager
        )
        
        assert result == "AI is a field of computer science..."
        
        # Verify search was performed
        self.mock_vector_store.search.assert_called_once_with(
            query="AI fundamentals",
            course_name="AI Course",
            lesson_number=None
        )
        
        # Verify sources were tracked
        sources = self.tool_manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["title"] == "AI Course - Lesson 1"
        assert sources[0]["link"] == "https://example.com/lesson1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])