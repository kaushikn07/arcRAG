"""
Agent Tools for NEXUS RAG System.
Provides semantic search, keyword search, graph search, SQL query, and calculation tools.
"""

import ast
import re
from dataclasses import dataclass
from typing import Optional

from retrieval.hybrid import hybrid_retrieve
from retrieval.keyword import bm25_search
from retrieval.graph import graph_retrieve
from retrieval.sql_retriever import sql_retrieve


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    output: str
    error: Optional[str] = None


class BaseTool:
    """Base class for all tools."""
    
    name: str = "base_tool"
    description: str = "Base tool description"
    
    async def run(self, input_str: str) -> ToolResult:
        raise NotImplementedError


class SemanticSearchTool(BaseTool):
    """Hybrid semantic search combining dense vector and keyword retrieval."""
    
    name = "semantic_search"
    description = (
        "Performs hybrid semantic search combining dense vector embeddings and keyword matching. "
        "Use this for general queries about document content, concepts, or when you need contextually relevant passages."
    )
    
    async def run(self, input_str: str) -> ToolResult:
        try:
            from config import settings
            from ingestion.embedder import embed_single
            
            # Generate embedding for the query
            query_embedding = await embed_single(input_str)
            
            # Run hybrid retrieval
            results = await hybrid_retrieve(input_str, query_embedding, top_k=5)
            
            if not results:
                return ToolResult(
                    success=True,
                    output="No relevant documents found for this query.",
                    error=None
                )
            
            # Format results as numbered context blocks
            formatted_parts = []
            for i, result in enumerate(results, 1):
                block = f"[{i}] Score: {result.score:.4f} | Doc: {result.doc_id} | Section: {result.section_id}\n"
                block += f"Text: {result.text}\n"
                if result.parent_text:
                    block += f"Parent Context: {result.parent_text}\n"
                formatted_parts.append(block)
            
            output = "\n---\n".join(formatted_parts)
            return ToolResult(success=True, output=output, error=None)
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Semantic search failed: {str(e)}"
            )


class KeywordSearchTool(BaseTool):
    """BM25 keyword-based full-text search."""
    
    name = "keyword_search"
    description = (
        "Performs BM25 keyword-based full-text search. "
        "Use this for queries containing specific terms, codes (e.g., ASC 842, IFRS 9), "
        "product names, or when exact phrase matching is important."
    )
    
    async def run(self, input_str: str) -> ToolResult:
        try:
            results = await bm25_search(input_str, top_k=5)
            
            if not results:
                return ToolResult(
                    success=True,
                    output="No documents found matching these keywords.",
                    error=None
                )
            
            # Format results
            formatted_parts = []
            for i, result in enumerate(results, 1):
                block = f"[{i}] Doc: {result.doc_id} | Section: {result.section_id}\n"
                block += f"Text: {result.text[:500]}{'...' if len(result.text) > 500 else ''}\n"
                formatted_parts.append(block)
            
            output = "\n---\n".join(formatted_parts)
            return ToolResult(success=True, output=output, error=None)
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Keyword search failed: {str(e)}"
            )


class GraphSearchTool(BaseTool):
    """Knowledge graph traversal for entity relationships."""
    
    name = "graph_search"
    description = (
        "Searches the knowledge graph for relationships between entities. "
        "Use this for queries about connections, relationships, suppliers, competitors, "
        "or how entities are related to each other."
    )
    
    async def run(self, input_str: str) -> ToolResult:
        try:
            # Extract entities from input using simple regex
            # Look for capitalized words/phrases that might be entities
            entity_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
            potential_entities = re.findall(entity_pattern, input_str)
            
            # Filter out common stop words and short words
            stop_words = {'The', 'This', 'That', 'These', 'Those', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For'}
            entities = [e for e in potential_entities if e not in stop_words and len(e) > 2]
            
            if not entities:
                return ToolResult(
                    success=True,
                    output="No recognizable entities found in the query for graph search.",
                    error=None
                )
            
            # Determine query type based on input
            query_type = "relational"
            if any(word in input_str.lower() for word in ['connected', 'related', 'relationship', 'link']):
                query_type = "relational"
            elif any(word in input_str.lower() for word in ['path', 'chain', 'through', 'via']):
                query_type = "multi_hop"
            
            results = await graph_retrieve(entities, query_type)
            
            if not results:
                return ToolResult(
                    success=True,
                    output=f"No relationships found in the knowledge graph for entities: {', '.join(entities)}",
                    error=None
                )
            
            # Format results
            output = "Graph Relationships Found:\n" + "\n".join(f"- {r}" for r in results)
            return ToolResult(success=True, output=output, error=None)
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Graph search failed: {str(e)}"
            )


class SQLQueryTool(BaseTool):
    """Natural language to SQL for structured financial metrics."""
    
    name = "sql_query"
    description = (
        "Executes SQL queries against structured financial metrics data. "
        "Use this for queries about specific numbers: revenue, earnings, EPS, growth rates, "
        "comparisons between companies, or time-series financial data."
    )
    
    async def run(self, input_str: str) -> ToolResult:
        try:
            # Extract entities to help with company filtering
            entity_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
            potential_entities = re.findall(entity_pattern, input_str)
            entities = [e for e in potential_entities if len(e) > 2]
            
            result = await sql_retrieve(input_str, entities)
            
            if not result or result == "No data found.":
                return ToolResult(
                    success=True,
                    output="No financial metrics data found matching this query.",
                    error=None
                )
            
            return ToolResult(success=True, output=result, error=None)
            
        except ValueError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"SQL query validation failed: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"SQL query failed: {str(e)}"
            )


class CalculatorTool(BaseTool):
    """Safe mathematical expression evaluator."""
    
    name = "calculator"
    description = (
        "Evaluates mathematical expressions safely. "
        "Use this for calculations like percentages, growth rates, ratios, or arithmetic operations. "
        "Supports +, -, *, /, //, ** (power), and parentheses."
    )
    
    async def run(self, input_str: str) -> ToolResult:
        try:
            # Clean the input
            expr = input_str.strip()
            
            # Validate the expression contains only safe characters
            allowed_pattern = r'^[\d\s\+\-\*\/\(\)\.\,\%]+$'
            if not re.match(allowed_pattern, expr):
                raise ValueError("Expression contains disallowed characters")
            
            # Replace commas with dots for decimal handling
            expr = expr.replace(',', '.')
            
            # Handle percentage signs
            expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', expr)
            
            # Parse and evaluate safely using ast
            tree = ast.parse(expr, mode='eval')
            
            # Validate the AST contains only safe nodes
            allowed_nodes = (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow,
                ast.USub, ast.UAdd, ast.Constant, ast.Load
            )
            
            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes):
                    raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
            
            # Evaluate the expression
            result = eval(compile(tree, '<string>', 'eval'))
            
            return ToolResult(
                success=True,
                output=f"Result: {result}",
                error=None
            )
            
        except ValueError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid expression: {str(e)}. Only simple math (+, -, *, /, //, **, parentheses) is allowed."
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Calculation failed: {str(e)}"
            )


# List of all available tools
ALL_TOOLS = [
    SemanticSearchTool(),
    KeywordSearchTool(),
    GraphSearchTool(),
    SQLQueryTool(),
    CalculatorTool(),
]

TOOL_REGISTRY = {tool.name: tool for tool in ALL_TOOLS}
