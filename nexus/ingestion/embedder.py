"""
Embedder for NEXUS

Generates embeddings using sentence-transformers (local).
"""

from typing import List, Optional

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import settings


class Embedder:
    """
    Generate embeddings using local sentence-transformers model.
    
    Uses the configured embedding model (default: all-MiniLM-L6-v2)
    which produces 384-dimensional vectors.
    """

    BATCH_SIZE = 128

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name or settings.embedding_model
        self.device = device
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            List of 384-dim embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.BATCH_SIZE), desc="Embedding", disable=not show_progress):
            batch = texts[i:i + self.BATCH_SIZE]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            all_embeddings.extend(batch_embeddings.tolist())
        
        return all_embeddings

    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            384-dim embedding vector
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()


# Global embedder instance
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Get the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def embed_texts(texts: List[str], **kwargs) -> List[List[float]]:
    """
    Convenience function to embed texts.
    
    Args:
        texts: List of texts to embed
        **kwargs: Additional arguments for Embedder
        
    Returns:
        List of embedding vectors
    """
    embedder = get_embedder()
    return embedder.embed_texts(texts, **kwargs)


def embed_single(text: str) -> List[float]:
    """
    Convenience function to embed a single text.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector
    """
    embedder = get_embedder()
    return embedder.embed_single(text)
