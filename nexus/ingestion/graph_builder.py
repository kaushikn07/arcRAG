"""
Graph Builder for NEXUS

Builds knowledge graph in Neo4j from document content using LLM-based entity extraction.
"""

import asyncio
import json
import re
from typing import Any

import httpx
from loguru import logger
from neo4j import AsyncGraphDatabase, AsyncDriver

from config import settings
from ingestion.parser import ParsedSection, ParsedDocument


class GraphBuilder:
    """
    Build and query knowledge graph in Neo4j.
    
    Extracts entities (Company, Person, Product, Market, RiskFactor, Regulation, Competitor)
    and relationships (ACQUIRED, PARTNERED_WITH, COMPETES_WITH, MENTIONED_IN, SUBJECT_TO)
    from financial documents using LLM.
    """

    VALID_ENTITY_TYPES = {
        "Company", "Person", "Product", "Market", "RiskFactor", "Regulation", "Competitor"
    }
    VALID_RELATION_TYPES = {
        "ACQUIRED", "PARTNERED_WITH", "COMPETES_WITH", "MENTIONED_IN", "SUBJECT_TO"
    }

    def __init__(self):
        """Initialize the graph builder."""
        self.driver: AsyncDriver | None = None
        self._semaphore = asyncio.Semaphore(10)
        self._connected = False
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password

    async def connect(self) -> bool:
        """Connect to Neo4j database."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=20
            )
            await self.driver.verify_connectivity()
            self._connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._connected = False
            return False

    async def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            await self.driver.close()
            self._connected = False
            logger.info("Neo4j connection closed")

    async def _ensure_constraints(self):
        """Create uniqueness constraints for entity types."""
        if not self._connected:
            return

        entity_types = ["Company", "Person", "Product", "Market", "RiskFactor", "Regulation", "Competitor"]
        
        async with self.driver.session() as session:
            for etype in entity_types:
                try:
                    cypher = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{etype}) REQUIRE (n.name) IS UNIQUE"
                    await session.run(cypher)
                except Exception as e:
                    logger.warning(f"Could not create constraint for {etype}: {e}")

    async def extract_entities(self, section_text: str, section_id: str, doc_id: str) -> dict[str, list]:
        """Extract named entities from section text using LLM.
        
        Args:
            section_text: The text content of the section
            section_id: Unique identifier for the section
            doc_id: Unique identifier for the document
            
        Returns:
            Dict with 'entities' and 'relations' lists
        """
        prompt = f"""Extract named entities from this financial document section.
Return ONLY a JSON object with two keys:
'entities': list of {{name, type, confidence}} where type is one of:
  Company, Person, Product, Market, RiskFactor, Regulation, Competitor
'relations': list of {{from_entity, relation_type, to_entity, confidence}} where
  relation_type is one of: ACQUIRED, PARTNERED_WITH, COMPETES_WITH, MENTIONED_IN, SUBJECT_TO
Only include entities with confidence >= 0.6. Normalize company names: strip Inc/Corp/Ltd.

Section text:
{section_text[:8000]}

Return ONLY the JSON object, no preamble."""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{settings.openrouter_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        "HTTP-Referer": "https://github.com/nexus-rag",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": settings.llm_model_fast,
                        "messages": [
                            {"role": "system", "content": "You are an entity extraction specialist. Return only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2000
                    }
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON, handling markdown code blocks
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                parsed = json.loads(content)
                
                # Validate and filter entities
                entities = []
                for ent in parsed.get("entities", []):
                    if ent.get("confidence", 0) >= 0.6 and ent.get("type") in self.VALID_ENTITY_TYPES:
                        name = ent.get("name", "")
                        if ent.get("type") == "Company":
                            name = re.sub(r"\s+(Inc|Corp|Corporation|Ltd|Limited|LLC)\.?$", "", name, flags=re.IGNORECASE)
                        entities.append({
                            "name": name,
                            "type": ent["type"],
                            "confidence": ent.get("confidence", 0.7)
                        })
                
                relations = []
                for rel in parsed.get("relations", []):
                    if rel.get("confidence", 0) >= 0.6 and rel.get("relation_type") in self.VALID_RELATION_TYPES:
                        relations.append({
                            "from_entity": rel["from_entity"],
                            "relation_type": rel["relation_type"],
                            "to_entity": rel["to_entity"],
                            "confidence": rel.get("confidence", 0.7)
                        })
                
                return {"entities": entities, "relations": relations}
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from entity extraction: {e}")
            return {"entities": [], "relations": []}
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": [], "relations": []}

    async def build_from_section(self, section: ParsedSection, doc_id: str, year: int) -> tuple[int, int]:
        """Build graph nodes and relationships from a single section.
        
        Args:
            section: The parsed section to process
            doc_id: Document identifier
            year: Document year
            
        Returns:
            Tuple of (nodes_created, relations_created)
        """
        if not self._connected:
            return (0, 0)

        async with self._semaphore:
            try:
                extracted = await self.extract_entities(section.raw_text, section.section_id, doc_id)
                
                if not extracted["entities"]:
                    return (0, 0)

                async with self.driver.session() as session:
                    nodes_created = 0
                    relations_created = 0
                    
                    for entity in extracted["entities"]:
                        cypher = f"""
                        MERGE (n:{entity['type']} {{name: $name}})
                        ON CREATE SET 
                            n.type = $type,
                            n.first_seen_doc = $doc_id,
                            n.first_seen_year = $year,
                            n.mention_count = 1
                        ON MATCH SET 
                            n.mention_count = coalesce(n.mention_count, 0) + 1
                        SET n.last_seen_doc = $doc_id,
                            n.last_seen_section = $section_id,
                            n.last_seen_year = $year
                        """
                        result = await session.run(
                            cypher,
                            name=entity["name"],
                            type=entity["type"],
                            doc_id=doc_id,
                            section_id=section.section_id,
                            year=year
                        )
                        summary = await result.consume()
                        nodes_created += summary.counters.nodes_created or 0
                    
                    for rel in extracted["relations"]:
                        from_type = "Entity"
                        to_type = "Entity"
                        for ent in extracted["entities"]:
                            if ent["name"] == rel["from_entity"]:
                                from_type = ent["type"]
                            if ent["name"] == rel["to_entity"]:
                                to_type = ent["type"]
                        
                        cypher = f"""
                        MATCH (a:{from_type} {{name: $from_name}})
                        MATCH (b:{to_type} {{name: $to_name}})
                        MERGE (a)-[r:{rel['relation_type']}]->(b)
                        ON CREATE SET 
                            r.first_seen_doc = $doc_id,
                            r.confidence = $confidence,
                            r.mention_count = 1
                        ON MATCH SET 
                            r.last_seen_doc = $doc_id,
                            r.mention_count = coalesce(r.mention_count, 0) + 1
                        """
                        result = await session.run(
                            cypher,
                            from_name=rel["from_entity"],
                            to_name=rel["to_entity"],
                            doc_id=doc_id,
                            confidence=rel["confidence"]
                        )
                        summary = await result.consume()
                        relations_created += summary.counters.relationships_created or 0
                    
                    return (nodes_created, relations_created)
                    
            except Exception as e:
                logger.error(f"Error building graph from section {section.section_id}: {e}")
                return (0, 0)

    async def build_from_document(self, doc: ParsedDocument, year: int) -> tuple[int, int]:
        """Build graph from all sections of a document.
        
        Args:
            doc: The parsed document to process
            year: Document year
            
        Returns:
            Tuple of (total_nodes_created, total_relations_created)
        """
        if not self._connected:
            logger.warning("Neo4j not connected, skipping graph build")
            return (0, 0)

        await self._ensure_constraints()
        
        tasks = [
            self.build_from_section(section, doc.doc_id, year)
            for section in doc.sections
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_nodes = 0
        total_relations = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Section processing failed: {result}")
            elif isinstance(result, tuple):
                nodes, relations = result
                total_nodes += nodes
                total_relations += relations
        
        logger.info(f"Graph build complete for {doc.doc_id}: {total_nodes} nodes, {total_relations} relations")
        return (total_nodes, total_relations)

    async def health_check(self) -> bool:
        """Check if Neo4j connection is healthy."""
        if not self._connected or not self.driver:
            return False
        try:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 AS ok")
                record = await result.single()
                return record is not None
        except Exception:
            return False


_graph_builder: GraphBuilder | None = None


async def get_graph_builder() -> GraphBuilder:
    """Get or create the global GraphBuilder instance."""
    global _graph_builder
    if _graph_builder is None:
        _graph_builder = GraphBuilder()
        await _graph_builder.connect()
    return _graph_builder


async def close_graph_builder():
    """Close the global GraphBuilder instance."""
    global _graph_builder
    if _graph_builder:
        await _graph_builder.close()
        _graph_builder = None
