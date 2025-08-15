import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Add backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator, ToolCallRound, MultiRoundState
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
    
    def test_generate_response_with_tools_no_tool_manager(self):
        """Test response generation with tools but no tool_manager (falls back to single call)"""
        mock_response = MockAnthropicResponse("Direct answer without tools")
        self.mock_client.messages.create.return_value = mock_response
        
        tools = [{"name": "search_course_content", "description": "Search courses"}]
        
        result = self.ai_generator.generate_response(
            "General question",
            tools=tools
            # Note: no tool_manager provided
        )
        
        assert result == "Direct answer without tools"
        
        # Verify tools were NOT passed to API (since no tool_manager)
        call_args = self.mock_client.messages.create.call_args[1]
        assert "tools" not in call_args  # No tools without tool_manager
        assert "tool_choice" not in call_args
    
    def test_generate_response_with_tools_and_tool_manager_no_tool_use(self):
        """Test response generation with tools and tool_manager available but no tool use"""
        mock_response = MockAnthropicResponse("Direct answer without using tools")
        self.mock_client.messages.create.return_value = mock_response
        
        tools = [{"name": "search_course_content", "description": "Search courses"}]
        mock_tool_manager = Mock()
        
        result = self.ai_generator.generate_response(
            "General question",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Direct answer without using tools"
        
        # Verify tools were passed to API (multi-round path)
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


class TestSequentialToolCalling:
    """Test sequential tool calling functionality (up to 2 rounds)"""
    
    def setup_method(self):
        """Setup test fixtures for sequential tool calling"""
        self.api_key = "test_api_key"
        self.model = "claude-sonnet-4-20250514"
        
        # Mock the Anthropic client
        self.mock_client = Mock()
        
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator(self.api_key, self.model)
    
    def test_single_round_completion(self):
        """Test that single round works when Claude doesn't need more tools"""
        # Single round: Tool use followed by direct answer
        round1_response = MockAnthropicResponse(
            content=[MockContentBlock(tool_use_data={
                "name": "search_course_content",
                "id": "tool_1",
                "input": {"query": "AI basics"}
            })],
            stop_reason="tool_use"
        )
        
        # Round 2: Direct answer (no more tools needed)
        final_response = MockAnthropicResponse(
            content="AI is the simulation of human intelligence in machines.",
            stop_reason="end_turn"
        )
        
        self.mock_client.messages.create.side_effect = [round1_response, final_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "AI content from search"
        
        result = self.ai_generator.generate_response(
            "What is AI?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        assert result == "AI is the simulation of human intelligence in machines."
        assert self.mock_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1
    
    def test_two_round_completion(self):
        """Test full 2-round sequential tool calling"""
        # Round 1: Get course outline
        round1_response = MockAnthropicResponse(
            content=[MockContentBlock(tool_use_data={
                "name": "get_course_outline",
                "id": "tool_1",
                "input": {"course_name": "Building Computer Use"}
            })],
            stop_reason="tool_use"
        )
        
        # Round 2: Search specific content based on outline
        round2_response = MockAnthropicResponse(
            content=[MockContentBlock(tool_use_data={
                "name": "search_course_content",
                "id": "tool_2",
                "input": {"query": "computer use agents", "course_name": "Building Computer Use"}
            })],
            stop_reason="tool_use"
        )
        
        # Final synthesis call (no tools)
        final_response = MockAnthropicResponse(
            content="The course covers computer-using AI agents that can autonomously navigate websites and perform tasks.",
            stop_reason="end_turn"
        )
        
        self.mock_client.messages.create.side_effect = [
            round1_response, round2_response, final_response
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline with lessons on computer use",
            "Content about AI agents performing computer tasks"
        ]
        
        result = self.ai_generator.generate_response(
            "What does the Computer Use course teach about AI agents?",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        assert "computer-using AI agents" in result
        assert self.mock_client.messages.create.call_count == 3  # 2 tool rounds + final synthesis
        assert mock_tool_manager.execute_tool.call_count == 2
    
    def test_max_rounds_reached(self):
        """Test behavior when maximum 2 rounds are reached"""
        # Both rounds use tools
        tool_response = MockAnthropicResponse(
            content=[MockContentBlock(tool_use_data={
                "name": "search_course_content",
                "id": "tool_x",
                "input": {"query": "search"}
            })],
            stop_reason="tool_use"
        )
        
        final_synthesis = MockAnthropicResponse(
            content="Final comprehensive answer based on multiple searches.",
            stop_reason="end_turn"
        )
        
        self.mock_client.messages.create.side_effect = [
            tool_response, tool_response, final_synthesis
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results"
        
        result = self.ai_generator.generate_response(
            "Complex multi-part question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        assert result == "Final comprehensive answer based on multiple searches."
        assert self.mock_client.messages.create.call_count == 3  # 2 tool rounds + final synthesis
        assert mock_tool_manager.execute_tool.call_count == 2
    
    def test_error_handling_in_round(self):
        """Test error handling during tool execution in a round"""
        round1_response = MockAnthropicResponse(
            content=[MockContentBlock(tool_use_data={
                "name": "search_course_content",
                "id": "tool_1",
                "input": {"query": "search"}
            })],
            stop_reason="tool_use"
        )
        
        self.mock_client.messages.create.side_effect = [
            round1_response, Exception("API Error in round 2")
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Results from round 1"
        
        result = self.ai_generator.generate_response(
            "Question that causes error",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Should return error message gracefully
        assert "error" in result.lower()
        assert self.mock_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1
    
    def test_tool_execution_error(self):
        """Test handling of tool execution errors"""
        round1_response = MockAnthropicResponse(
            content=[MockContentBlock(tool_use_data={
                "name": "search_course_content",
                "id": "tool_1",
                "input": {"query": "search"}
            })],
            stop_reason="tool_use"
        )
        
        self.mock_client.messages.create.side_effect = [round1_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        result = self.ai_generator.generate_response(
            "Question that causes tool error",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Should handle tool error gracefully
        assert "error" in result.lower()
        assert self.mock_client.messages.create.call_count == 1
        assert mock_tool_manager.execute_tool.call_count == 1
    
    def test_conversation_context_building(self):
        """Test that conversation context is properly built across rounds"""
        # Round 1: Tool use
        round1_response = MockAnthropicResponse(
            content=[MockContentBlock(tool_use_data={
                "name": "get_course_outline",
                "id": "tool_1",
                "input": {"course_name": "MCP"}
            })],
            stop_reason="tool_use"
        )
        
        # Round 2: Tool use with context
        round2_response = MockAnthropicResponse(
            content=[MockContentBlock(tool_use_data={
                "name": "search_course_content", 
                "id": "tool_2",
                "input": {"query": "protocol architecture", "course_name": "MCP"}
            })],
            stop_reason="tool_use"
        )
        
        # Final response
        final_response = MockAnthropicResponse(
            content="MCP is a protocol for AI context management with specific architecture.",
            stop_reason="end_turn"
        )
        
        self.mock_client.messages.create.side_effect = [
            round1_response, round2_response, final_response
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "MCP course outline with protocol lessons",
            "Detailed protocol architecture information"
        ]
        
        result = self.ai_generator.generate_response(
            "Explain MCP protocol architecture",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        assert "protocol for AI context management" in result
        
        # Verify conversation context was built properly
        # Check that second API call includes context from first round
        second_call_args = self.mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]
        
        # Should have: user query, assistant round1, user tool_results, 
        assert len(messages) >= 3
        assert messages[0]["role"] == "user"  # Original query
        assert messages[1]["role"] == "assistant"  # Round 1 response
        assert messages[2]["role"] == "user"  # Round 1 tool results
    
    def test_system_prompt_selection(self):
        """Test that multi-round system prompt is used for tool queries"""
        mock_response = MockAnthropicResponse("Test response")
        self.mock_client.messages.create.return_value = mock_response
        
        mock_tool_manager = Mock()
        
        self.ai_generator.generate_response(
            "Question requiring tools",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        call_args = self.mock_client.messages.create.call_args[1]
        system_prompt = call_args["system"]
        
        # Verify multi-round system prompt is used
        assert "You can make up to 2 rounds of tool calls" in system_prompt
        assert "Progressive refinement" in system_prompt
        assert "Multi-Round Tool Usage Guidelines" in system_prompt
        
    def test_backwards_compatibility_single_call(self):
        """Test that single calls without tools still work as before"""
        mock_response = MockAnthropicResponse("Direct response without tools")
        self.mock_client.messages.create.return_value = mock_response
        
        result = self.ai_generator.generate_response("What is AI?")
        
        assert result == "Direct response without tools"
        assert self.mock_client.messages.create.call_count == 1
        
        # Verify single-round system prompt is used
        call_args = self.mock_client.messages.create.call_args[1]
        system_prompt = call_args["system"]
        assert "One tool call per query maximum" in system_prompt
        assert "Multi-Round Tool Usage Guidelines" not in system_prompt


class TestMultiRoundDataStructures:
    """Test the data structures used for multi-round conversations"""
    
    def test_tool_call_round_creation(self):
        """Test ToolCallRound data structure"""
        round_data = ToolCallRound(
            round_number=1,
            api_response=Mock(),
            has_tool_calls=True
        )
        
        assert round_data.round_number == 1
        assert round_data.has_tool_calls is True
        assert round_data.tool_calls == []
        assert round_data.tool_results == []
        assert round_data.error_message is None
    
    def test_multi_round_state_properties(self):
        """Test MultiRoundState properties and methods"""
        state = MultiRoundState()
        
        # Test initial state
        assert state.current_round_number == 1
        assert state.has_remaining_rounds is True
        assert state.last_round_had_tools is False
        assert state.is_complete() is False
        
        # Add a round without tools
        round1 = ToolCallRound(round_number=1, api_response=Mock(), has_tool_calls=False)
        state.rounds.append(round1)
        
        assert state.current_round_number == 2
        assert state.is_complete() is True  # No tool calls = complete
        
        # Reset and test max rounds
        state.rounds = []
        round1 = ToolCallRound(round_number=1, api_response=Mock(), has_tool_calls=True)
        round2 = ToolCallRound(round_number=2, api_response=Mock(), has_tool_calls=True)
        state.rounds.extend([round1, round2])
        
        assert state.is_complete() is True  # Max rounds reached
        assert state.has_remaining_rounds is False
        
        # Test error termination
        state.rounds = []
        error_round = ToolCallRound(
            round_number=1, 
            api_response=Mock(), 
            has_tool_calls=False,
            error_message="Test error"
        )
        state.rounds.append(error_round)
        
        assert state.is_complete() is True  # Error = complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])