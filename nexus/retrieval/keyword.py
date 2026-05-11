"""
Keyword Retrieval for NEXUS

Retrieves chunks using full-text search (BM25-style).
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from db.neon import fetch
from config import settings


class KeywordRetriever:
    """
    Retrieve chunks using keyword/full-text search.
    
    Uses PostgreSQL's tsvector/tsquery with GIN index
    for efficient text search.
    """

    def __init__(self, top_k: int = None):
        """
        Initialize the keyword retriever.
        
        Args:
            top_k: Number of results to return
        """
        self.top_k = top_k or settings.top_k_keyword

    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks matching keywords in the query.
        
        Args:
            query: Query text
            filters: Optional filters (company, year, doc_type)
            top_k: Override default top_k
            
        Returns:
            List of chunk dicts with metadata and scores
        """
        k = top_k or self.top_k
        
        # Convert query to tsquery format
        tsquery = self._build_tsquery(query)
        
        # Build SQL with optional filters
        where_clauses = ["c.text_search @@ $1::tsquery"]
        params = [tsquery]
        param_idx = 2
        
        if filters:
            if filters.get('company'):
                where_clauses.append(f"d.company = ${param_idx}")
                params.append(filters['company'])
                param_idx += 1
            
            if filters.get('year'):
                where_clauses.append(f"d.year = ${param_idx}")
                params.append(filters['year'])
                param_idx += 1
            
            if filters.get('doc_type'):
                where_clauses.append(f"d.doc_type = ${param_idx}")
                params.append(filters['doc_type'])
                param_idx += 1
        
        where_clause = " AND ".join(where_clauses)
        
        sql = f"""
        SELECT 
            c.chunk_id,
            c.text,
            c.token_count,
            c.page_number,
            d.doc_id,
            d.company,
            d.year,
            d.doc_type,
            s.section_id,
            s.heading,
            ts_rank(c.text_search, $1::tsquery) AS relevance_score
        FROM chunks c
        JOIN documents d ON c.doc_id = d.doc_id
        LEFT JOIN sections s ON c.section_id = s.section_id
        WHERE {where_clause}
        ORDER BY ts_rank(c.text_search, $1::tsquery) DESC
        LIMIT ${param_idx}
        """
        
        params.append(k)
        
        try:
            results = await fetch(sql, *params)
            
            return [
                {
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "token_count": r["token_count"],
                    "page_number": r["page_number"],
                    "doc_id": r["doc_id"],
                    "company": r["company"],
                    "year": r["year"],
                    "doc_type": r["doc_type"],
                    "section_id": r["section_id"],
                    "heading": r["heading"],
                    "relevance_score": float(r["relevance_score"]),
                    "retrieval_method": "keyword"
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Keyword retrieval error: {e}")
            raise

    def _build_tsquery(self, query: str) -> str:
        """
        Build a tsquery from the input query.
        
        Uses & (AND) between significant terms and | (OR) for variants.
        """
        # Simple tokenization - split on whitespace and punctuation
        import re
        tokens = re.findall(r'\w+', query.lower())
        
        # Filter out very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        
        if not tokens:
            return "''"
        
        # Use prefix matching for better recall
        prefixed = [f"{t}:*" for t in tokens[:10]]  # Limit to 10 terms
        
        # Join with AND
        return " & ".join(prefixed)

    async def retrieve_with_highlights(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = None,
        highlight_radius: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with highlighted matching snippets.
        
        Args:
            query: Query text
            filters: Optional filters
            top_k: Number of results
            highlight_radius: Characters around match to show
            
        Returns:
            List of chunk dicts with highlights
        """
        results = await self.retrieve(query, filters, top_k)
        
        # Add highlights to each result
        query_terms = query.lower().split()
        
        for result in results:
            text = result["text"]
            text_lower = text.lower()
            
            # Find first matching term position
            match_pos = len(text)
            for term in query_terms:
                if len(term) > 2:
                    pos = text_lower.find(term)
                    if pos != -1 and pos < match_pos:
                        match_pos = pos
            
            # Extract snippet with context
            start = max(0, match_pos - highlight_radius)
            end = min(len(text), match_pos + len(query_terms[0]) + highlight_radius)
            
            snippet = text[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            
            result["snippet"] = snippet
            result["match_position"] = match_pos
        
        return results


# Global retriever instance
_keyword_retriever: Optional[KeywordRetriever] = None


def get_keyword_retriever() -> KeywordRetriever:
    """Get the global keyword retriever instance."""
    global _keyword_retriever
    if _keyword_retriever is None:
        _keyword_retriever = KeywordRetriever()
    return _keyword_retriever


async def retrieve_keyword(
    query: str,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function for keyword retrieval.
    
    Args:
        query: Query text
        **kwargs: Additional arguments
        
    Returns:
        List of chunk dicts
    """
    retriever = get_keyword_retriever()
    return await retriever.retrieve(query, **kwargs)
