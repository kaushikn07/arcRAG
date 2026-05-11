"""
Graph Builder for NEXUS

Builds knowledge graph in Neo4j from document content.
"""

from typing import List, Optional, Dict, Any
from loguru import logger

try:
    from neo4j import GraphDatabase, AsyncGraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    logger.warning("neo4j not installed. Install with: pip install neo4j")

from config import settings


class GraphBuilder:
    """
    Build and query knowledge graph in Neo4j.
    
    Creates nodes for:
    - Documents
    - Companies
    - Financial Metrics
    - Sections/Chunks
    
    And relationships between them.
    """

    def __init__(self):
        """Initialize the graph builder."""
        self._driver = None
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password

    async def connect(self) -> None:
        """Connect to Neo4j database."""
        if not HAS_NEO4J:
            raise ImportError(
                "neo4j not installed. Install with: pip install neo4j"
            )
        
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            
            # Test connection
            try:
                await self._driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self.uri}")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise

    async def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    async def create_document_node(
        self,
        doc_id: int,
        company: str,
        year: int,
        doc_type: str,
        filename: str,
        **kwargs
    ) -> None:
        """Create a document node in the graph."""
        query = """
        MERGE (d:Document {doc_id: $doc_id})
        SET d.company = $company,
            d.year = $year,
            d.doc_type = $doc_type,
            d.filename = $filename,
            d.page_count = $page_count,
            d.ingested_at = datetime($ingested_at)
        """
        
        await self.execute(query, {
            "doc_id": doc_id,
            "company": company,
            "year": year,
            "doc_type": doc_type,
            "filename": filename,
            "page_count": kwargs.get('page_count', 0),
            "ingested_at": kwargs.get('ingested_at', '')
        })

    async def create_company_node(
        self,
        company_name: str,
        **properties
    ) -> None:
        """Create or update a company node."""
        query = """
        MERGE (c:Company {name: $name})
        SET c += $properties
        """
        
        await self.execute(query, {
            "name": company_name,
            "properties": properties
        })

    async def create_metric_node(
        self,
        metric_id: int,
        company: str,
        year: int,
        metric_name: str,
        value: float,
        unit: str,
        period: Optional[str] = None,
        doc_id: Optional[int] = None
    ) -> None:
        """Create a financial metric node."""
        query = """
        MERGE (m:FinancialMetric {metric_id: $metric_id})
        SET m.company = $company,
            m.year = $year,
            m.metric_name = $metric_name,
            m.value = $value,
            m.unit = $unit,
            m.period = $period
        """
        
        await self.execute(query, {
            "metric_id": metric_id,
            "company": company,
            "year": year,
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "period": period
        })
        
        # Create relationships
        if doc_id:
            rel_query = """
            MATCH (m:FinancialMetric {metric_id: $metric_id})
            MATCH (d:Document {doc_id: $doc_id})
            MERGE (m)-[:FROM_DOCUMENT]->(d)
            """
            await self.execute(rel_query, {"metric_id": metric_id, "doc_id": doc_id})
        
        # Link to company
        company_rel_query = """
        MATCH (m:FinancialMetric {metric_id: $metric_id})
        MATCH (c:Company {name: $company})
        MERGE (m)-[:BELONGS_TO_COMPANY]->(c)
        """
        await self.execute(company_rel_query, {"metric_id": metric_id, "company": company})

    async def create_section_relationships(
        self,
        doc_id: int,
        sections: List[Dict[str, Any]]
    ) -> None:
        """Create section nodes and link to document."""
        for section in sections:
            query = """
            MERGE (s:Section {section_id: $section_id})
            SET s.heading = $heading,
                s.heading_level = $heading_level,
                s.sequence_order = $sequence_order,
                s.content_type = $content_type,
                s.token_count = $token_count
            
            WITH s
            MATCH (d:Document {doc_id: $doc_id})
            MERGE (s)-[:PART_OF_DOCUMENT]->(d)
            """
            
            await self.execute(query, {
                "section_id": section.get('section_id'),
                "heading": section.get('heading'),
                "heading_level": section.get('heading_level', 1),
                "sequence_order": section.get('sequence_order', 0),
                "content_type": section.get('content_type', 'text'),
                "token_count": section.get('token_count', 0),
                "doc_id": doc_id
            })

    async def execute(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query."""
        if self._driver is None:
            await self.connect()
        
        async with self._driver.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def query_metrics_by_company(
        self,
        company: str,
        year: Optional[int] = None,
        metric_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Query financial metrics for a company."""
        conditions = ["(c:Company {name: $company})"]
        params = {"company": company}
        
        if year:
            conditions.append("m.year = $year")
            params["year"] = year
        
        if metric_names:
            conditions.append("m.metric_name IN $metric_names")
            params["metric_names"] = metric_names
        
        where_clause = " AND ".join([f"{c}" for c in conditions])
        
        query = f"""
        MATCH (c:Company)<-[:BELONGS_TO_COMPANY]-(m:FinancialMetric)
        WHERE c.name = $company
        """
        
        if year:
            query += " AND m.year = $year"
        
        if metric_names:
            query += " AND m.metric_name IN $metric_names"
        
        query += """
        RETURN m.metric_name AS name, 
               m.value AS value, 
               m.unit AS unit,
               m.year AS year,
               m.period AS period
        ORDER BY m.year, m.metric_name
        """
        
        return await self.execute(query, params)

    async def get_company_relationships(
        self,
        company: str
    ) -> List[Dict[str, Any]]:
        """Get all relationships for a company."""
        query = """
        MATCH (c:Company {name: $company})-[r]-(connected)
        RETURN c.name AS company, 
               type(r) AS relationship,
               labels(connected) AS connected_types,
               properties(connected) AS connected_properties
        """
        
        return await self.execute(query, {"company": company})

    async def run_health_check(self) -> bool:
        """Check if Neo4j is healthy."""
        try:
            result = await self.execute("RETURN 1 AS ok")
            return len(result) > 0
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False


# Global graph builder instance
_graph_builder: Optional[GraphBuilder] = None


def get_graph_builder() -> GraphBuilder:
    """Get the global graph builder instance."""
    global _graph_builder
    if _graph_builder is None:
        _graph_builder = GraphBuilder()
    return _graph_builder


async def init_graph() -> GraphBuilder:
    """Initialize and connect the graph builder."""
    builder = get_graph_builder()
    await builder.connect()
    return builder


async def close_graph() -> None:
    """Close the graph connection."""
    builder = get_graph_builder()
    await builder.close()
