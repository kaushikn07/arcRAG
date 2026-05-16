"""
Ingestion Orchestrator for NEXUS

Coordinates the full document ingestion pipeline.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from db.neon import execute, fetchrow, get_db
from ingestion.parser import PDFParser, ParsedDocument, parse_document
from ingestion.chunker import TextChunker, Chunk, chunk_document
from ingestion.embedder import Embedder, embed_texts
from ingestion.hyde import HyDEGenerator, generate_hypothetical_questions
from ingestion.metrics_extractor import MetricsExtractor, extract_financial_metrics
from ingestion.graph_builder import GraphBuilder, init_graph
from config import settings


class IngestionOrchestrator:
    """
    Orchestrate the full document ingestion pipeline.
    
    Pipeline stages:
    1. Parse PDF to sections
    2. Chunk sections into retrievable units
    3. Generate embeddings for chunks
    4. Generate hypothetical questions (HyDE)
    5. Extract financial metrics
    6. Store everything in database
    7. Build knowledge graph in Neo4j
    """

    def __init__(self):
        """Initialize the orchestrator."""
        self.parser = PDFParser()
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.hyde_generator = HyDEGenerator()
        self.metrics_extractor = MetricsExtractor()
        self.graph_builder = GraphBuilder()

    async def ingest_document(
        self,
        file_path: str,
        company: str,
        year: int,
        doc_type: str,
        run_hyde: bool = True,
        run_metrics: bool = True,
        run_graph: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest a single document through the full pipeline.
        
        Args:
            file_path: Path to the PDF file
            company: Company name
            year: Fiscal year
            doc_type: Document type (e.g., '10-K', '10-Q')
            run_hyde: Whether to generate hypothetical questions
            run_metrics: Whether to extract financial metrics
            run_graph: Whether to build knowledge graph
            
        Returns:
            Dict with ingestion statistics
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stats = {
            "filename": path.name,
            "company": company,
            "year": year,
            "doc_type": doc_type,
            "stages_completed": [],
            "sections_count": 0,
            "chunks_count": 0,
            "questions_count": 0,
            "metrics_count": 0
        }

        try:
            logger.info(f"Starting ingestion of {file_path}")

            # Stage 1: Register document in database
            doc_id = await self._register_document(path.name, company, year, doc_type)
            stats["doc_id"] = doc_id

            # Stage 2: Parse PDF
            parsed = self.parser.parse(file_path)
            parsed.company = company
            parsed.year = year
            parsed.doc_type = doc_type
            
            section_ids = await self._store_sections(doc_id, parsed.sections)
            stats["sections_count"] = len(parsed.sections)
            stats["stages_completed"].append("parsed")
            logger.info(f"Parsed {len(parsed.sections)} sections")

            # Stage 3: Chunk sections
            all_chunks = []
            for i, section in enumerate(parsed.sections):
                chunks = self.chunker.chunk_text(
                    text=section.raw_text,
                    section_id=section_ids[i],
                    doc_id=doc_id,
                    page_number=section.page_number
                )
                all_chunks.extend(chunks)

            stats["chunks_count"] = len(all_chunks)
            stats["stages_completed"].append("chunked")
            logger.info(f"Created {len(all_chunks)} chunks")

            # Stage 4: Generate embeddings and store chunks
            await self._store_chunks_with_embeddings(all_chunks)
            stats["stages_completed"].append("embedded")
            logger.info("Generated and stored embeddings")

            # Stage 5: Generate hypothetical questions (HyDE)
            if run_hyde:
                question_count = await self._generate_and_store_hyde(
                    all_chunks, doc_id
                )
                stats["questions_count"] = question_count
                stats["stages_completed"].append("hyde")
                logger.info(f"Generated {question_count} hypothetical questions")

            # Stage 6: Extract financial metrics
            if run_metrics:
                metric_count = await self._extract_and_store_metrics(
                    parsed, doc_id
                )
                stats["metrics_count"] = metric_count
                stats["stages_completed"].append("metrics")
                logger.info(f"Extracted {metric_count} financial metrics")

            # Stage 7: Update document stage
            await self._update_document_stage(doc_id, "indexed")

            # Stage 8: Build knowledge graph
            if run_graph:
                await self._build_graph(doc_id, company, year, parsed, all_chunks)
                stats["stages_completed"].append("graph")
                logger.info("Built knowledge graph")

            logger.info(f"Ingestion complete: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            # Update stage to indicate failure
            if "doc_id" in stats:
                await self._update_document_stage(stats["doc_id"], f"failed:{e}")
            raise

    async def _register_document(
        self,
        filename: str,
        company: str,
        year: int,
        doc_type: str
    ) -> int:
        """Register document in database and return doc_id."""
        query = """
        INSERT INTO documents (company, year, doc_type, filename, stage_completed)
        VALUES ($1, $2, $3, $4, 'uploaded')
        RETURNING doc_id
        """
        result = await fetchrow(query, company, year, doc_type, filename)
        return result["doc_id"]

    async def _store_sections(
        self,
        doc_id: int,
        sections: List[Any]
    ) -> List[int]:
        """Store sections and return their IDs."""
        section_ids = []
        
        for section in sections:
            query = """
            INSERT INTO sections (doc_id, heading, heading_level, sequence_order, 
                                  content_type, raw_text, token_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING section_id
            """
            result = await fetchrow(
                query, doc_id, section.heading, section.heading_level,
                section.sequence_order, section.content_type,
                section.raw_text, self.chunker.count_tokens(section.raw_text)
            )
            section_ids.append(result["section_id"])
        
        return section_ids

    async def _store_chunks_with_embeddings(self, chunks: List[Chunk]) -> None:
        """Store chunks with their embeddings."""
        if not chunks:
            return

        # Batch embed all chunks
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts, show_progress=True)

        # Store in batches
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            args_list = []
            for chunk, emb in zip(batch, batch_embeddings):
                args_list.append((
                    chunk.doc_id,
                    chunk.section_id,
                    chunk.chunk_level,
                    chunk.parent_chunk_id,
                    chunk.text,
                    chunk.token_count,
                    emb.tolist(),
                    chunk.page_number
                ))

            query = """
            INSERT INTO chunks (doc_id, section_id, chunk_level, parent_chunk_id,
                               text, token_count, embedding, page_number)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            
            db = get_db()
            await db.executemany(query, args_list)

    async def _generate_and_store_hyde(
        self,
        chunks: List[Chunk],
        doc_id: int
    ) -> int:
        """Generate and store hypothetical questions."""
        total_questions = 0

        # Sample chunks for HyDE (don't do all for efficiency)
        sample_chunks = chunks[::3]  # Every 3rd chunk

        for chunk in sample_chunks[:50]:  # Max 50 chunks
            questions = await self.hyde_generator.generate_questions(
                chunk_text=chunk.text,
                num_questions=3,
                chunk_id=None,  # Will set after insert
                doc_id=doc_id
            )

            for q in questions:
                # Generate embedding for question
                q_embedding = self.embedder.embed_query(q.question_text)

                # We need chunk_id first - will update after chunk lookup
                # For now, store with placeholder and update
                pass

        # Simplified: store questions linked to doc
        for chunk in sample_chunks[:50]:
            questions = await self.hyde_generator.generate_questions(
                chunk_text=chunk.text,
                num_questions=2,
                doc_id=doc_id
            )
            
            for q in questions:
                q_emb = self.embedder.embed_query(q.question_text)
                
                # Find chunk_id by text match (simplified)
                query = """
                SELECT chunk_id FROM chunks 
                WHERE doc_id = $1 AND text = $2 
                LIMIT 1
                """
                result = await fetchrow(query, doc_id, chunk.text)
                chunk_id = result["chunk_id"] if result else None

                insert_query = """
                INSERT INTO hypothetical_questions (chunk_id, doc_id, question_text, embedding)
                VALUES ($1, $2, $3, $4)
                """
                await execute(insert_query, chunk_id, doc_id, q.question_text, q_emb.tolist())
                total_questions += 1

        return total_questions

    async def _extract_and_store_metrics(
        self,
        parsed: ParsedDocument,
        doc_id: int
    ) -> int:
        """Extract and store financial metrics."""
        total_metrics = 0

        # Combine all section text for extraction
        full_text = "\n\n".join([s.raw_text for s in parsed.sections])

        metrics = await self.metrics_extractor.extract_metrics(
            text=full_text,
            company=parsed.company,
            year=parsed.year
        )

        for metric in metrics:
            query = """
            INSERT INTO financial_metrics 
            (doc_id, company, year, period, metric_name, value, unit, source_section_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            await execute(
                query, doc_id, metric.company, metric.year, metric.period,
                metric.metric_name, metric.value, metric.unit, metric.source_section_id
            )
            total_metrics += 1

        return total_metrics

    async def _update_document_stage(self, doc_id: int, stage: str) -> None:
        """Update document's stage_completed field."""
        query = """
        UPDATE documents SET stage_completed = $1 WHERE doc_id = $2
        """
        await execute(query, stage, doc_id)

    async def _build_graph(
        self,
        doc_id: int,
        company: str,
        year: int,
        parsed: ParsedDocument,
        chunks: List[Chunk]
    ) -> None:
        """Build knowledge graph in Neo4j."""
        try:
            await self.graph_builder.connect()

            # Create company node
            await self.graph_builder.create_company_node(company)

            # Create document node
            await self.graph_builder.create_document_node(
                doc_id=doc_id,
                company=company,
                year=year,
                doc_type=parsed.doc_type,
                filename=parsed.filename,
                page_count=parsed.page_count
            )

            # Create section nodes
            sections_data = []
            for i, section in enumerate(parsed.sections):
                sections_data.append({
                    "section_id": i + 1,  # Simplified
                    "heading": section.heading,
                    "heading_level": section.heading_level,
                    "sequence_order": section.sequence_order,
                    "content_type": section.content_type,
                    "token_count": section.token_count or 0
                })

            await self.graph_builder.create_section_relationships(doc_id, sections_data)

        except Exception as e:
            logger.warning(f"Graph building skipped: {e}")
        finally:
            await self.graph_builder.close()


# Global orchestrator instance
_orchestrator: Optional[IngestionOrchestrator] = None


def get_orchestrator() -> IngestionOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = IngestionOrchestrator()
    return _orchestrator


async def ingest_file(
    file_path: str,
    company: str,
    year: int,
    doc_type: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to ingest a file.
    
    Args:
        file_path: Path to PDF
        company: Company name
        year: Fiscal year
        doc_type: Document type
        **kwargs: Additional arguments
        
    Returns:
        Ingestion statistics
    """
    orchestrator = get_orchestrator()
    return await orchestrator.ingest_document(file_path, company, year, doc_type, **kwargs)
