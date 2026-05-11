"""
SQL-based Retrieval for NEXUS

Retrieves structured financial metrics using SQL queries.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from db.neon import fetch, fetchrow


class SQLRetriever:
    """
    Retrieve structured data using SQL queries.
    
    Used for precise financial metric lookups
    that don't require semantic search.
    """

    async def get_metrics(
        self,
        company: str,
        year: Optional[int] = None,
        period: Optional[str] = None,
        metric_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get financial metrics by criteria.
        
        Args:
            company: Company name
            year: Optional year filter
            period: Optional period (Q1, Q2, FY, etc.)
            metric_names: Optional list of specific metrics
            
        Returns:
            List of metric dicts
        """
        conditions = ["company = $1"]
        params = [company]
        param_idx = 2
        
        if year:
            conditions.append(f"year = ${param_idx}")
            params.append(year)
            param_idx += 1
        
        if period:
            conditions.append(f"period = ${param_idx}")
            params.append(period)
            param_idx += 1
        
        if metric_names:
            placeholders = ", ".join([f"${param_idx + i}" for i in range(len(metric_names))])
            conditions.append(f"metric_name IN ({placeholders})")
            params.extend(metric_names)
            param_idx += len(metric_names)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        SELECT metric_id, company, year, period, metric_name, 
               value, unit, source_section_id
        FROM financial_metrics
        WHERE {where_clause}
        ORDER BY year DESC, period, metric_name
        """
        
        results = await fetch(query, *params)
        
        return [
            {
                "metric_id": r["metric_id"],
                "company": r["company"],
                "year": r["year"],
                "period": r["period"],
                "metric_name": r["metric_name"],
                "value": float(r["value"]),
                "unit": r["unit"],
                "source_section_id": r["source_section_id"]
            }
            for r in results
        ]

    async def get_metric_trend(
        self,
        company: str,
        metric_name: str,
        years: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get trend data for a specific metric across years.
        
        Args:
            company: Company name
            metric_name: Metric to track
            years: Optional list of years
            
        Returns:
            List of yearly values
        """
        conditions = ["company = $1", "metric_name = $2"]
        params = [company, metric_name]
        param_idx = 3
        
        if years:
            placeholders = ", ".join([f"${param_idx + i}" for i in range(len(years))])
            conditions.append(f"year IN ({placeholders})")
            params.extend(years)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        SELECT year, period, value, unit
        FROM financial_metrics
        WHERE {where_clause}
        ORDER BY year, period
        """
        
        results = await fetch(query, *params)
        
        return [
            {
                "year": r["year"],
                "period": r["period"],
                "value": float(r["value"]),
                "unit": r["unit"]
            }
            for r in results
        ]

    async def compare_companies(
        self,
        companies: List[str],
        metric_name: str,
        year: int
    ) -> List[Dict[str, Any]]:
        """
        Compare a metric across multiple companies.
        
        Args:
            companies: List of company names
            metric_name: Metric to compare
            year: Year to compare
            
        Returns:
            List of company-value pairs
        """
        placeholders = ", ".join([f"${i + 1}" for i in range(len(companies))])
        
        query = f"""
        SELECT company, value, unit, period
        FROM financial_metrics
        WHERE company IN ({placeholders})
          AND metric_name = ${len(companies) + 1}
          AND year = ${len(companies) + 2}
        ORDER BY value DESC
        """
        
        params = companies + [metric_name, year]
        results = await fetch(query, *params)
        
        return [
            {
                "company": r["company"],
                "value": float(r["value"]),
                "unit": r["unit"],
                "period": r["period"]
            }
            for r in results
        ]

    async def get_document_stats(self, doc_id: int) -> Dict[str, Any]:
        """
        Get statistics for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Stats dict with counts and info
        """
        # Get document info
        doc_query = """
        SELECT doc_id, company, year, doc_type, filename, 
               page_count, ingested_at, stage_completed
        FROM documents
        WHERE doc_id = $1
        """
        doc = await fetchrow(doc_query, doc_id)
        
        if not doc:
            return {}
        
        # Get chunk count
        chunk_query = """
        SELECT COUNT(*) as chunk_count, SUM(token_count) as total_tokens
        FROM chunks
        WHERE doc_id = $1
        """
        chunk_stats = await fetchrow(chunk_query, doc_id)
        
        # Get metric count
        metric_query = """
        SELECT COUNT(*) as metric_count
        FROM financial_metrics
        WHERE doc_id = $1
        """
        metric_stats = await fetchrow(metric_query, doc_id)
        
        return {
            "document": dict(doc),
            "chunks": {
                "count": chunk_stats["chunk_count"],
                "total_tokens": chunk_stats["total_tokens"] or 0
            },
            "metrics": {
                "count": metric_stats["metric_count"]
            }
        }


# Global retriever instance
_sql_retriever: Optional[SQLRetriever] = None


def get_sql_retriever() -> SQLRetriever:
    """Get the global SQL retriever instance."""
    global _sql_retriever
    if _sql_retriever is None:
        _sql_retriever = SQLRetriever()
    return _sql_retriever


async def retrieve_metrics_sql(
    company: str,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function for SQL-based metric retrieval.
    
    Args:
        company: Company name
        **kwargs: Additional arguments
        
    Returns:
        List of metric dicts
    """
    retriever = get_sql_retriever()
    return await retriever.get_metrics(company, **kwargs)
