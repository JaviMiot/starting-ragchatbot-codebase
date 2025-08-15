import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ToolCallRound:
    """Represents a single round of tool calling"""
    round_number: int
    api_response: Any  # Anthropic response object
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    has_tool_calls: bool = False
    error_message: Optional[str] = None


@dataclass
class MultiRoundState:
    """Manages state across multiple tool calling rounds"""
    rounds: List[ToolCallRound] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    base_params: Dict[str, Any] = field(default_factory=dict)
    tools: Optional[List] = None
    tool_manager: Any = None
    max_rounds: int = 2
    
    @property
    def current_round_number(self) -> int:
        return len(self.rounds) + 1
    
    @property
    def has_remaining_rounds(self) -> bool:
        return len(self.rounds) < self.max_rounds
    
    @property
    def last_round_had_tools(self) -> bool:
        return bool(self.rounds and self.rounds[-1].has_tool_calls)
    
    def is_complete(self) -> bool:
        """Check if conversation should terminate"""
        if not self.rounds:
            return False
        
        last_round = self.rounds[-1]
        
        # Terminate if max rounds reached
        if len(self.rounds) >= self.max_rounds:
            return True
        
        # Terminate if last round had no tool calls (Claude provided final answer)
        if not last_round.has_tool_calls:
            return True
        
        # Terminate if last round had an error
        if last_round.error_message:
            return True
        
        return False


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # System prompt for single-round tool usage (backwards compatibility)
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search tools for course information.

Tool Usage Guidelines:
- **Course outline queries**: Use the get_course_outline tool for questions about course structure, lesson lists, or course overviews
- **Content search queries**: Use the search_course_content tool for questions about specific course content or detailed educational materials
- **One tool call per query maximum**
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Course Outline Responses:
When using the course outline tool, ensure your response includes:
- Course title and course link
- Complete numbered lesson list with lesson titles
- Any additional course metadata (instructor, etc.)

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline tool first, then answer
- **Course content questions**: Use search_course_content tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    # Enhanced system prompt for multi-round tool usage
    MULTI_ROUND_SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search tools for course information.

Multi-Round Tool Usage Guidelines:
- **You can make up to 2 rounds of tool calls** to gather comprehensive information
- **Round 1**: Start with broad searches, course outline retrieval, or initial exploration
- **Round 2**: Use Round 1 results to make more targeted, specific searches based on what you learned
- **Progressive refinement**: Each round should build on previous results to create comprehensive responses
- **Strategic tool selection**: Choose tools based on what you discovered from previous rounds

Effective Tool Usage Patterns:
- **Course outline first**: Use get_course_outline to understand course structure, then search_course_content for specific details
- **Broad then narrow**: Start with general content searches, then focus on specific lessons/topics found
- **Cross-reference verification**: Use multiple searches to verify, expand, or compare information across courses
- **Multi-part questions**: Break down complex queries into sequential tool calls

Course Outline Responses:
When using the course outline tool, ensure your response includes:
- Course title and course link
- Complete numbered lesson list with lesson titles
- Any additional course metadata (instructor, etc.)

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Use tools strategically across multiple rounds for comprehensive answers
- **Complex queries**: Leverage multiple rounds to gather complete information before responding
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or round descriptions
 - Do not mention "based on the search results," "using tools," or "in the first round"
 - Synthesize ALL tool results from ALL rounds into a unified response

All responses must be:
1. **Comprehensive** - Use multiple rounds to gather complete information when needed
2. **Accurate** - Base responses on actual tool results from all rounds combined
3. **Educational** - Maintain instructional value across synthesized information
4. **Clear** - Use accessible language that integrates information seamlessly
5. **Example-supported** - Include relevant examples from any round when they aid understanding
Provide only the direct answer to what was asked, synthesizing all gathered information.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 sequential rounds of tool calling for complex queries.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Route to appropriate response generation method
        if tools and tool_manager:
            return self._generate_with_multi_round_tools(query, conversation_history, tools, tool_manager)
        else:
            return self._generate_single_call(query, conversation_history)
    
    def _generate_single_call(self, query: str, conversation_history: Optional[str] = None) -> str:
        """Generate response without tools (single API call)"""
        
        # Build system content efficiently
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        return response.content[0].text
    
    def _generate_with_multi_round_tools(self, query: str, conversation_history: Optional[str], 
                                       tools: List, tool_manager) -> str:
        """
        Generate response with up to 2 sequential rounds of tool calling.
        
        Each round allows Claude to reason about previous tool results before
        making additional tool calls for more comprehensive responses.
        """
        
        # Initialize multi-round state
        state = self._initialize_multi_round_state(query, conversation_history, tools, tool_manager)
        
        try:
            # Execute up to 2 rounds of tool calling
            while state.has_remaining_rounds and not state.is_complete():
                round_result = self._execute_tool_round(state)
                state.rounds.append(round_result)
                
                # Update conversation context with round results
                if round_result.has_tool_calls and round_result.tool_results:
                    self._update_state_with_round_results(state, round_result)
                elif round_result.error_message:
                    # Error occurred - terminate with graceful message
                    break
            
            # Extract final response from last round or generate synthesis
            return self._extract_final_response(state)
            
        except Exception as e:
            # Graceful error handling
            print(f"Error in multi-round tool execution: {e}")
            if state.rounds:
                return "I found some information but encountered an error completing your request."
            else:
                return "I'm unable to search for that information right now."
    
    def _initialize_multi_round_state(self, query: str, conversation_history: Optional[str], 
                                    tools: List, tool_manager) -> MultiRoundState:
        """Initialize state for multi-round conversation"""
        
        # Build system content with multi-round prompt
        system_content = (
            f"{self.MULTI_ROUND_SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.MULTI_ROUND_SYSTEM_PROMPT
        )
        
        # Prepare base parameters for API calls
        base_params = {
            **self.base_params,
            "system": system_content
        }
        
        # Initialize conversation with user query
        initial_messages = [{"role": "user", "content": query}]
        
        return MultiRoundState(
            rounds=[],
            messages=initial_messages,
            base_params=base_params,
            tools=tools,
            tool_manager=tool_manager,
            max_rounds=2
        )
    
    def _execute_tool_round(self, state: MultiRoundState) -> ToolCallRound:
        """Execute a single round of tool calling"""
        
        round_number = state.current_round_number
        
        try:
            # Prepare API call parameters for this round
            api_params = {
                **state.base_params,
                "messages": state.messages.copy(),
                "tools": state.tools,
                "tool_choice": {"type": "auto"}
            }
            
            # Make API call
            response = self.client.messages.create(**api_params)
            
            # Create round result
            round_result = ToolCallRound(
                round_number=round_number,
                api_response=response,
                has_tool_calls=(response.stop_reason == "tool_use")
            )
            
            # Execute tools if present
            if round_result.has_tool_calls:
                try:
                    tool_calls, tool_results = self._execute_tools_in_response(response, state.tool_manager)
                    round_result.tool_calls = tool_calls
                    round_result.tool_results = tool_results
                except Exception as tool_error:
                    round_result.error_message = f"Tool execution error: {str(tool_error)}"
                    round_result.has_tool_calls = False
            
            return round_result
            
        except Exception as api_error:
            # Handle API errors gracefully
            return ToolCallRound(
                round_number=round_number,
                api_response=None,
                error_message=f"API error in round {round_number}: {str(api_error)}"
            )
    
    def _execute_tools_in_response(self, response, tool_manager) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute all tool calls in a response and return calls and results"""
        
        tool_calls = []
        tool_results = []
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                # Track tool call
                tool_call = {
                    "name": content_block.name,
                    "id": content_block.id,
                    "input": content_block.input
                }
                tool_calls.append(tool_call)
                
                # Execute tool
                tool_result_content = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                # Format tool result for API
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result_content
                }
                tool_results.append(tool_result)
        
        return tool_calls, tool_results
    
    def _update_state_with_round_results(self, state: MultiRoundState, round_result: ToolCallRound):
        """Update conversation state with results from a completed round"""
        
        # Add Claude's response with tool calls to message history
        state.messages.append({
            "role": "assistant", 
            "content": round_result.api_response.content
        })
        
        # Add tool results as user message if any were generated
        if round_result.tool_results:
            state.messages.append({
                "role": "user", 
                "content": round_result.tool_results
            })
    
    def _extract_final_response(self, state: MultiRoundState) -> str:
        """Extract final response text from the conversation state"""
        
        if not state.rounds:
            return "I was unable to process your request."
        
        last_round = state.rounds[-1]
        
        # If last round had an error, return error message
        if last_round.error_message:
            if len(state.rounds) > 1:
                return "I found some information but encountered an error completing your search."
            else:
                return "I encountered an error while searching for information."
        
        # If last round had no tool calls, extract text response
        if not last_round.has_tool_calls and last_round.api_response:
            return last_round.api_response.content[0].text
        
        # If we completed max rounds with tool calls, generate final synthesis
        if len(state.rounds) >= state.max_rounds and last_round.has_tool_calls:
            return self._generate_final_synthesis(state)
        
        # Fallback - shouldn't normally reach here
        return "I was unable to complete your request."
    
    def _generate_final_synthesis(self, state: MultiRoundState) -> str:
        """Generate final response synthesis without tools after max rounds"""
        
        try:
            # Prepare final API call without tools
            final_params = {
                **state.base_params,
                "messages": state.messages.copy()
                # Note: explicitly no tools parameter for final synthesis
            }
            
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
            
        except Exception as e:
            print(f"Error generating final synthesis: {e}")
            return "I gathered information from multiple searches but had trouble summarizing it."
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text