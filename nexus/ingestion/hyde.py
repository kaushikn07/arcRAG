"""
HyDE (Hypothetical Document Embeddings) for NEXUS

Generates hypothetical questions from chunks to improve retrieval.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

import httpx
from config import settings


@dataclass
class HypotheticalQuestion:
    """Represents a HyDE-generated question."""
    question_text: str
    chunk_id: Optional[int] = None
    doc_id: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HyDEGenerator:
    """
    Generate hypothetical questions from text chunks using LLM.
    
    HyDE improves retrieval by generating questions that the chunk
    could answer, then embedding those questions for similarity search.
    """

    def __init__(self, model: str = None):
        """
        Initialize HyDE generator.
        
        Args:
            model: LLM model to use for generation
        """
        self.model = model or settings.llm_model_fast
        self.base_url = settings.openrouter_base_url
        self.api_key = settings.openrouter_api_key

    async def generate_questions(
        self,
        chunk_text: str,
        num_questions: int = 3,
        chunk_id: Optional[int] = None,
        doc_id: Optional[int] = None
    ) -> List[HypotheticalQuestion]:
        """
        Generate hypothetical questions from a chunk.
        
        Args:
            chunk_text: Text of the chunk
            num_questions: Number of questions to generate
            chunk_id: Optional chunk identifier
            doc_id: Optional document identifier
            
        Returns:
            List of HypotheticalQuestion objects
        """
        if not chunk_text.strip():
            return []

        prompt = self._build_prompt(chunk_text, num_questions)
        
        try:
            response_text = await self._call_llm(prompt)
            questions = self._parse_questions(response_text)
            
            return [
                HypotheticalQuestion(
                    question_text=q,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    metadata={"source_chunk_length": len(chunk_text)}
                )
                for q in questions[:num_questions]
            ]
            
        except Exception as e:
            logger.error(f"Error generating HyDE questions: {e}")
            return []

    async def generate_questions_batch(
        self,
        chunks: List[Dict[str, Any]],
        num_questions_per_chunk: int = 3
    ) -> List[HypotheticalQuestion]:
        """
        Generate questions for multiple chunks.
        
        Args:
            chunks: List of dicts with 'text', 'chunk_id', 'doc_id' keys
            num_questions_per_chunk: Questions per chunk
            
        Returns:
            List of all HypotheticalQuestion objects
        """
        all_questions = []
        
        for chunk in chunks:
            questions = await self.generate_questions(
                chunk_text=chunk.get('text', ''),
                num_questions=num_questions_per_chunk,
                chunk_id=chunk.get('chunk_id'),
                doc_id=chunk.get('doc_id')
            )
            all_questions.extend(questions)
        
        logger.info(f"Generated {len(all_questions)} hypothetical questions")
        return all_questions

    def _build_prompt(self, text: str, num_questions: int) -> str:
        """Build the prompt for question generation."""
        # Truncate if too long
        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        prompt = f"""You are an expert at generating relevant questions from financial document excerpts.

Given the following text excerpt from a financial document, generate exactly {num_questions} distinct questions that this text could answer.

Focus on:
- Financial metrics and figures mentioned
- Company performance indicators
- Strategic initiatives or business developments
- Risk factors or challenges
- Forward-looking statements

Text excerpt:
{text}

Generate exactly {num_questions} questions, one per line, without numbering or bullets. Only output the questions, nothing else.

Questions:"""
        
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nexus-rag",
            "X-Title": "NEXUS"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        url = f"{self.base_url}/chat/completions"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def _parse_questions(self, text: str) -> List[str]:
        """Parse generated questions from LLM response."""
        questions = []
        
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Remove numbering/bullets
            line = line.lstrip('1234567890.-*•')
            line = line.strip()
            
            if line and line.endswith('?'):
                questions.append(line)
        
        # If parsing failed, try to extract any sentence with question mark
        if not questions:
            import re
            questions = re.findall(r'[^.!?]+\?', text)
        
        return questions


# Global HyDE generator instance
_hyde_generator: Optional[HyDEGenerator] = None


def get_hyde_generator() -> HyDEGenerator:
    """Get the global HyDE generator instance."""
    global _hyde_generator
    if _hyde_generator is None:
        _hyde_generator = HyDEGenerator()
    return _hyde_generator


async def generate_hypothetical_questions(
    text: str,
    num_questions: int = 3,
    **kwargs
) -> List[HypotheticalQuestion]:
    """
    Convenience function to generate hypothetical questions.
    
    Args:
        text: Text to generate questions from
        num_questions: Number of questions to generate
        **kwargs: Additional arguments
        
    Returns:
        List of HypotheticalQuestion objects
    """
    generator = get_hyde_generator()
    return await generator.generate_questions(text, num_questions, **kwargs)
