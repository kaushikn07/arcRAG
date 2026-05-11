"""
Graph Retrieval for NEXUS

Retrieves information from Neo4j knowledge graph.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ingestion.graph_builder import GraphBuilder


class GraphRetriever:
    """
    Retrieve information from the Neo4j knowledge graph.
    
    Supports queries like:
    - Get all metrics for a company
    - Compare metrics across years
    - Find related documents
    """

    def __init__(self):
        """Initialize the graph retriever."""
        self.graph = GraphBuilder()

    async def retrieve_metrics(
        self,
        company: str,
        year: Optional[int] = None,
        metric_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve financial metrics from the graph.
        
        Args:
            company: Company name
            year: Optional year filter
            metric_names: Optional list of specific metrics
            
        Returns:
            List of metric dicts
        """
        await self.graph.connect()
        
        try:
            results = await self.graph.query_metrics_by_company(
                company, year, metric_names
            )
            
            return [
                {
                    "name": r["name"],
                    "value": float(r["value"]),
                    "unit": r["unit"],
                    "year": r["year"],
                    "period": r["period"],
                    "source": "graph"
                }
                for r in results
            ]
        finally:
            await self.graph.close()

    async def retrieve_company_info(self, company: str) -> Dict[str, Any]:
        """
        Retrieve company information and relationships.
        
        Args:
            company: Company name
            
        Returns:
            Company info dict with relationships
        """
        await self.graph.connect()
        
        try:
            relationships = await self.graph.get_company_relationships(company)
            
            # Aggregate by relationship type
            by_type = {}
            for rel in relationships:
                rel_type = rel["relationship"]
                if rel_type not in by_type:
                    by_type[rel_type] = []
                by_type[rel_type].append(rel["connected_properties"])
            
            return {
                "company": company,
                "relationships": by_type,
                "document_count": len(by_type.get("PART_OF_DOCUMENT", [])),
                "metric_count": len(by_type.get("BELONGS_TO_COMPANY", []))
            }
        finally:
            await self.graph.close()

    async def search_documents(
        self,
        company: Optional[str] = None,
        year: Optional[int] = None,
        doc_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents in the graph.
        
        Args:
            company: Optional company filter
            year: Optional year filter
            doc_type: Optional document type filter
            
        Returns:
            List of document dicts
        """
        await self.graph.connect()
        
        conditions = []
        params = {}
        
        if company:
            conditions.append("d.company = $company")
            params["company"] = company
        
        if year:
            conditions.append("d.year = $year")
            params["year"] = year
        
        if doc_type:
            conditions.append("d.doc_type = $doc_type")
            params["doc_type"] = doc_type
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
        MATCH (d:Document)
        {where_clause}
        RETURN d.doc_id AS doc_id,
               d.company AS company,
               d.year AS year,
               d.doc_type AS doc_type,
               d.filename AS filename,
               d.page_count AS page_count
        ORDER BY d.year DESC, d.doc_id
        """
        
        try:
            results = await self.graph.execute(query, params if params else None)
            return results
        finally:
            await self.graph.close()


# Global retriever instance
_graph_retriever: Optional[GraphRetriever] = None


def get_graph_retriever() -> GraphRetriever:
    """Get the global graph retriever instance."""
    global _graph_retriever
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _graph_retriever


async def retrieve_from_graph(
    company: str,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function for graph retrieval.
    
    Args:
        company: Company name
        **kwargs: Additional arguments
        
    Returns:
        List of results
    """
    retriever = get_graph_retriever()
    return await retriever.retrieve_metrics(company, **kwargs)
