"""
Embedder for NEXUS

Generates embeddings using sentence-transformers (local).
"""

from typing import List, Optional, Union
import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

from config import settings


class Embedder:
    """
    Generate embeddings using local sentence-transformers model.
    
    Uses the configured embedding model (default: all-MiniLM-L6-v2)
    which produces 384-dimensional vectors.
    """

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
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Embedding model loaded successfully")
        
        return self._model

    def embed(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings (recommended for cosine similarity)
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (n_texts x embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([]).reshape(0, settings.embedding_dim)
        
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress
            )
            
            # Ensure correct dimension
            if embeddings.shape[1] != settings.embedding_dim:
                logger.warning(
                    f"Expected embedding dim {settings.embedding_dim}, "
                    f"got {embeddings.shape[1]}"
                )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            1D numpy array of embedding
        """
        embeddings = self.embed([query], normalize=True)
        return embeddings[0]

    def embed_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple documents in batches.
        
        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_embeddings = self.embed(batch, show_progress=False)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)

    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        top_k: int = 5
    ) -> tuple:
        """
        Calculate cosine similarity and return top-k matches.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Matrix of document embeddings
            top_k: Number of top results to return
            
        Returns:
            Tuple of (indices, scores) for top-k matches
        """
        # Normalize if not already normalized
        if np.linalg.norm(query_embedding) != 1.0:
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if not np.allclose(np.linalg.norm(document_embeddings, axis=1), 1.0):
            norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
            document_embeddings = document_embeddings / (norms + 1e-9)
        
        # Calculate cosine similarity
        similarities = document_embeddings @ query_embedding
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores


# Global embedder instance
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Get the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def embed_texts(texts: List[str], **kwargs) -> np.ndarray:
    """
    Convenience function to embed texts.
    
    Args:
        texts: List of texts to embed
        **kwargs: Additional arguments for Embedder
        
    Returns:
        numpy array of embeddings
    """
    embedder = get_embedder()
    return embedder.embed(texts, **kwargs)


def embed_query(query: str, **kwargs) -> np.ndarray:
    """
    Convenience function to embed a query.
    
    Args:
        query: Query text
        **kwargs: Additional arguments for Embedder
        
    Returns:
        1D numpy array of embedding
    """
    embedder = get_embedder()
    return embedder.embed_query(query, **kwargs)
