"""HyDE (Hypothetical Document Embeddings) question generator."""

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Optional

import httpx
from loguru import logger

from config import settings
from ingestion.chunker import Chunk


@dataclass
class HypotheticalQuestion:
    """Represents a HyDE-generated question."""
    q_id: str
    chunk_id: str
    doc_id: str
    question_text: str
    embedding: Optional[list[float]] = None


class HyDEGenerator:
    """Generate hypothetical questions for chunks using LLM."""

    def __init__(self):
        self.base_url = settings.openrouter_base_url.rstrip("/")
        self.api_key = settings.openrouter_api_key
        self.model = settings.llm_model_fast
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/nexus-rag",
            "Content-Type": "application/json",
        }
        # Rate limit: max 10 concurrent requests
        self.semaphore = asyncio.Semaphore(10)

    async def _call_llm(self, prompt: str) -> str:
        """Call OpenRouter API with the given prompt."""
        async with self.semaphore:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 512,
                }
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error from OpenRouter: {e.response.status_code} - {e.response.text}")
                    raise
                except Exception as e:
                    logger.error(f"Error calling OpenRouter: {e}")
                    raise

    async def generate_hyde_questions(self, chunk_text: str, chunk_id: str) -> list[str]:
        """
        Generate 3-5 hypothetical questions that the chunk text answers.

        Args:
            chunk_text: The text of the chunk
            chunk_id: The ID of the chunk

        Returns:
            List of question strings (each ending with '?')
        """
        prompt = (
            "Generate 3-5 specific questions that this text passage directly answers. "
            "Return only a JSON array of question strings. No preamble.\n\n"
            f"Text passage:\n{chunk_text}"
        )

        try:
            response_text = await self._call_llm(prompt)

            # Try to parse JSON array from response
            # Handle cases where response might have markdown code blocks
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            questions = json.loads(response_text)

            if not isinstance(questions, list):
                logger.warning(f"Expected JSON array, got {type(questions)}: {response_text[:100]}")
                return []

            # Filter: keep only strings that end with '?'
            valid_questions = []
            for q in questions:
                if isinstance(q, str) and q.strip().endswith("?"):
                    valid_questions.append(q.strip())
                elif isinstance(q, str):
                    # Add '?' if missing
                    valid_questions.append(q.strip() + "?")

            return valid_questions

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}. Response: {response_text[:200]}")
            return []
        except Exception as e:
            logger.error(f"Error generating HyDE questions: {e}")
            return []

    async def generate_and_store_all(self, chunks: list[Chunk]) -> int:
        """
        Generate HyDE questions for all chunks and store in database.

        Args:
            chunks: List of Chunk objects

        Returns:
            Total number of questions stored
        """
        from db.neon import execute

        total_stored = 0

        # Process in batches of 20 to avoid overwhelming the system
        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            tasks = []

            for chunk in batch:
                task = self._process_chunk(chunk)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Store results
            insert_queries = []
            insert_args = []

            for chunk, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing chunk {chunk.chunk_id}: {result}")
                    continue

                questions = result
                for question_text in questions:
                    q_id = hashlib.sha256(f"{chunk.chunk_id}:{question_text}".encode()).hexdigest()[:16]
                    insert_queries.append(
                        """
                        INSERT INTO hypothetical_questions (q_id, chunk_id, doc_id, question_text, embedding)
                        VALUES ($1, $2, $3, $4, NULL)
                        ON CONFLICT (q_id) DO NOTHING
                        """
                    )
                    insert_args.append((q_id, chunk.chunk_id, chunk.doc_id, question_text))

            # Batch insert
            if insert_queries:
                # Use executemany-style approach
                for query, args in zip(insert_queries, insert_args):
                    try:
                        await execute(query, *args)
                        total_stored += 1
                    except Exception as e:
                        logger.error(f"Failed to store question {args[0]}: {e}")

            logger.info(f"Processed batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")

        return total_stored

    async def _process_chunk(self, chunk: Chunk) -> list[str]:
        """Process a single chunk and return its questions."""
        return await self.generate_hyde_questions(chunk.text, chunk.chunk_id)


# Convenience function
async def generate_hyde_questions(chunk_text: str, chunk_id: str) -> list[str]:
    """Generate HyDE questions for a chunk."""
    generator = HyDEGenerator()
    return await generator.generate_hyde_questions(chunk_text, chunk_id)
