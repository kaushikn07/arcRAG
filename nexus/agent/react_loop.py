"""
ReAct Agent Loop for NEXUS RAG System.
Implements reasoning with action loops, tool execution, and response synthesis.
"""

import re
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple

import httpx
from loguru import logger

from config import settings
from agent.tools import ToolResult, TOOL_REGISTRY
from agent.query_plan import QueryPlan


@dataclass
class TraceStep:
    """A single step in the ReAct reasoning trace."""
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResponse:
    """Final response from the agent."""
    final_answer: str
    trace: List[TraceStep] = field(default_factory=list)
    total_steps: int = 0
    tools_used: List[str] = field(default_factory=list)


class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent for financial document intelligence.
    
    Implements a loop where the agent:
    1. Thinks about what to do next
    2. Selects and executes a tool
    3. Observes the result
    4. Repeats until it has enough information or reaches limits
    """
    
    # Token budget constants
    MAX_CONTEXT_TOKENS = 4096
    TOKEN_BUDGET_THRESHOLD = 0.8  # 80% of max triggers synthesis
    
    # Regex patterns for parsing LLM output
    THOUGHT_PATTERN = r'Thought:\s*(.+?)(?=Action:|$)'
    ACTION_PATTERN = r'Action:\s*(\w+)'
    ACTION_INPUT_PATTERN = r'Action Input:\s*(.+?)(?=Observation:|Thought:|$)'
    FINAL_ANSWER_PATTERN = r'(?:Final Answer:|Conclusion:)\s*(.+)'
    
    def __init__(self, tools: Optional[List] = None, max_steps: int = 8):
        self.max_steps = max_steps
        self.tools = tools if tools else list(TOOL_REGISTRY.values())
        self.tool_names = {tool.name for tool in self.tools}
        
        # Track conversation history for context management
        self._context_history: List[str] = []
    
    async def run(self, query: str, query_plan: QueryPlan) -> AgentResponse:
        """
        Execute the ReAct loop for a given query and plan.
        
        Args:
            query: The user's natural language query
            query_plan: Pre-computed query plan with routing decisions
            
        Returns:
            AgentResponse with final answer and execution trace
        """
        # Handle out-of-scope queries immediately
        if query_plan.query_type == 'out_of_scope':
            return AgentResponse(
                final_answer="This question is outside the scope of the financial documents available.",
                trace=[],
                total_steps=0,
                tools_used=[]
            )
        
        # Filter tools based on query plan requirements
        active_tools = self._filter_tools_by_plan(query_plan)
        
        if not active_tools:
            # Default to semantic search if no tools activated
            active_tools = [TOOL_REGISTRY['semantic_search']]
        
        # Initialize tracking variables
        trace: List[TraceStep] = []
        tools_used: Set[str] = set()
        action_history: Set[Tuple[str, str]] = set()  # For loop detection
        context_tokens = 0
        
        # Build system prompt with tool descriptions
        system_prompt = self._build_system_prompt(active_tools)
        
        # Initialize conversation state
        current_query = query
        graph_fallback_triggered = False
        
        for step_num in range(1, self.max_steps + 1):
            logger.info(f"ReAct Step {step_num}/{self.max_steps}")
            
            # Check token budget
            if context_tokens > self.MAX_CONTEXT_TOKENS * self.TOKEN_BUDGET_THRESHOLD:
                # Force synthesis due to token budget
                final_answer = await self._synthesize_answer(
                    query=query,
                    trace=trace,
                    query_plan=query_plan
                )
                return AgentResponse(
                    final_answer=f"Context is getting long. {final_answer}",
                    trace=trace,
                    total_steps=step_num,
                    tools_used=list(tools_used)
                )
            
            # Get LLM response for this step
            llm_response = await self._get_llm_step(
                system_prompt=system_prompt,
                query=current_query,
                trace=trace,
                step_num=step_num
            )
            
            # Parse the LLM response
            thought, action, action_input = self._parse_llm_response(llm_response)
            
            if not action:
                # LLM provided a final answer directly
                final_answer = thought or llm_response
                return AgentResponse(
                    final_answer=final_answer,
                    trace=trace,
                    total_steps=step_num,
                    tools_used=list(tools_used)
                )
            
            # Check for action loops
            action_key = (action.lower(), action_input.lower().strip())
            if action_key in action_history:
                # Inject warning about repeating actions
                observation = (
                    "You are repeating a previous action. "
                    "Try a different approach or conclude with Final Answer."
                )
            else:
                action_history.add(action_key)
                
                # Execute the tool
                observation = await self._execute_tool(
                    action=action,
                    action_input=action_input,
                    query_plan=query_plan,
                    graph_fallback_triggered=graph_fallback_triggered
                )
                
                # Handle graph-to-vector fallback
                if action == 'graph_search' and 'No relationships found' in observation:
                    graph_fallback_triggered = True
                    # Automatically retry with semantic search
                    logger.info("Graph search returned empty, triggering semantic search fallback")
                    observation = await self._execute_tool(
                        action='semantic_search',
                        action_input=action_input,
                        query_plan=query_plan,
                        graph_fallback_triggered=False
                    )
                    action = 'semantic_search (fallback from graph_search)'
                
                tools_used.add(action.split()[0])  # Handle fallback notation
            
            # Record this step
            trace_step = TraceStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation
            )
            trace.append(trace_step)
            
            # Update context tokens estimate (rough estimate: 4 chars ≈ 1 token)
            context_tokens += len(observation) // 4
            
            # Prepare for next iteration
            current_query = self._build_next_query(query, trace_step)
        
        # Max steps reached
        final_answer = await self._synthesize_answer(
            query=query,
            trace=trace,
            query_plan=query_plan
        )
        
        return AgentResponse(
            final_answer=f"Reached reasoning limit. {final_answer}",
            trace=trace,
            total_steps=self.max_steps,
            tools_used=list(tools_used)
        )
    
    def _filter_tools_by_plan(self, query_plan: QueryPlan) -> List:
        """Filter available tools based on query plan requirements."""
        active_tools = []
        
        if query_plan.requires_graph:
            if 'graph_search' in TOOL_REGISTRY:
                active_tools.append(TOOL_REGISTRY['graph_search'])
        
        if query_plan.requires_sql:
            if 'sql_query' in TOOL_REGISTRY:
                active_tools.append(TOOL_REGISTRY['sql_query'])
        
        if query_plan.requires_vector or query_plan.requires_keyword:
            if query_plan.keyword_heavy and 'keyword_search' in TOOL_REGISTRY:
                active_tools.append(TOOL_REGISTRY['keyword_search'])
            elif 'semantic_search' in TOOL_REGISTRY:
                active_tools.append(TOOL_REGISTRY['semantic_search'])
        
        # Always include calculator for metric queries
        if query_plan.query_type == 'metric':
            if 'calculator' in TOOL_REGISTRY:
                active_tools.append(TOOL_REGISTRY['calculator'])
        
        return active_tools
    
    def _build_system_prompt(self, active_tools: List) -> str:
        """Build the system prompt with tool descriptions and format instructions."""
        tool_descriptions = "\n\n".join([
            f"Tool: {tool.name}\nDescription: {tool.description}"
            for tool in active_tools
        ])
        
        tool_names = ", ".join([tool.name for tool in active_tools])
        
        return f"""You are an AI assistant specialized in analyzing financial documents. 
You have access to the following tools:

{tool_descriptions}

Available tools: {tool_names}

You must reason step-by-step before taking any action. Follow this exact format:

Thought: <Your reasoning about what to do next>
Action: <tool_name from available tools>
Action Input: <the input to pass to the tool>

After executing the action, you will receive an Observation. Continue reasoning until you have enough information.

When you have gathered sufficient information, provide your final answer:

Final Answer: <Your comprehensive answer to the original query>

Important rules:
1. Only use the tools listed above
2. Each Action must be followed by Action Input
3. Think carefully before choosing which tool to use
4. If a tool returns no results, try a different approach
5. Synthesize information from multiple sources when possible
6. Be precise with numbers and cite sources when available
"""
    
    async def _get_llm_step(
        self,
        system_prompt: str,
        query: str,
        trace: List[TraceStep],
        step_num: int
    ) -> str:
        """Get the next step from the LLM."""
        # Build conversation history
        conversation = self._build_conversation_history(query, trace, step_num)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{settings.openrouter_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        "HTTP-Referer": "https://github.com/nexus-rag",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": settings.llm_model_classify,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": conversation}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
                
            except Exception as e:
                logger.error(f"LLM call failed at step {step_num}: {e}")
                return f"Thought: Error calling LLM. I should synthesize what I know.\nFinal Answer: Unable to complete analysis due to technical error: {str(e)}"
    
    def _build_conversation_history(
        self,
        query: str,
        trace: List[TraceStep],
        step_num: int
    ) -> str:
        """Build the conversation history for the LLM."""
        parts = [f"Original Query: {query}\n"]
        
        if trace:
            parts.append("\nPrevious Steps:\n")
            for i, step in enumerate(trace, 1):
                parts.append(f"Step {i}:\n")
                parts.append(f"  Thought: {step.thought}\n")
                parts.append(f"  Action: {step.action}\n")
                parts.append(f"  Action Input: {step.action_input}\n")
                parts.append(f"  Observation: {step.observation}\n\n")
        
        parts.append(f"\nCurrent Step: {step_num}\n")
        parts.append("What is your next Thought and Action? ")
        
        return "".join(parts)
    
    def _parse_llm_response(self, llm_output: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse Thought, Action, and Action Input from LLM output."""
        thought_match = re.search(self.THOUGHT_PATTERN, llm_output, re.DOTALL | re.IGNORECASE)
        action_match = re.search(self.ACTION_PATTERN, llm_output, re.IGNORECASE)
        action_input_match = re.search(self.ACTION_INPUT_PATTERN, llm_output, re.DOTALL | re.IGNORECASE)
        
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        action_input = action_input_match.group(1).strip() if action_input_match else None
        
        return thought, action, action_input
    
    async def _execute_tool(
        self,
        action: str,
        action_input: str,
        query_plan: QueryPlan,
        graph_fallback_triggered: bool
    ) -> str:
        """Execute a tool and return its observation."""
        # Normalize action name (handle fallback notation)
        action_name = action.split()[0].lower()
        
        if action_name not in TOOL_REGISTRY:
            return f"Error: Unknown tool '{action_name}'. Available tools: {', '.join(self.tool_names)}"
        
        tool = TOOL_REGISTRY[action_name]
        
        try:
            result: ToolResult = await tool.run(action_input)
            
            if result.success:
                return result.output
            else:
                return f"Tool error: {result.error}"
                
        except Exception as e:
            return f"Tool execution failed: {str(e)}"
    
    def _build_next_query(self, original_query: str, last_step: TraceStep) -> str:
        """Build the query for the next iteration."""
        return f"{original_query}\n\nLatest observation: {last_step.observation}\n\nWhat should I do next?"
    
    async def _synthesize_answer(
        self,
        query: str,
        trace: List[TraceStep],
        query_plan: QueryPlan
    ) -> str:
        """Synthesize a final answer from the accumulated trace."""
        # Build context from trace
        context_parts = []
        for step in trace:
            if step.observation and step.observation.strip():
                context_parts.append(f"[{step.action}] {step.observation[:500]}")
        
        context = "\n\n".join(context_parts) if context_parts else "No additional context gathered."
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{settings.openrouter_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        "HTTP-Referer": "https://github.com/nexus-rag",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": settings.llm_model_synthesis,
                        "messages": [
                            {"role": "system", "content": (
                                "You are a financial analyst synthesizing information from multiple sources. "
                                "Provide a clear, accurate, and well-structured answer. "
                                "Cite specific data points when available. "
                                "If information is incomplete, acknowledge limitations."
                            )},
                            {"role": "user", "content": (
                                f"Original Query: {query}\n\n"
                                f"Query Type: {query_plan.query_type}\n\n"
                                f"Gathered Context:\n{context}\n\n"
                                "Please synthesize a comprehensive final answer based on the context above."
                            )}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 800
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
                
            except Exception as e:
                logger.error(f"Synthesis LLM call failed: {e}")
                # Fallback: manually synthesize from trace
                return self._manual_synthesize(query, trace)
    
    def _manual_synthesize(self, query: str, trace: List[TraceStep]) -> str:
        """Manual synthesis fallback when LLM is unavailable."""
        if not trace:
            return "Unable to gather information to answer this query."
        
        observations = [step.observation for step in trace if step.observation]
        
        if not observations:
            return "Tools were executed but no useful information was found."
        
        return f"Based on the analysis:\n\n" + "\n\n".join(observations[:3])
