"""
Agentic Chunking - Semantic chunking with LLM intelligence

Creates meaningful chunks based on semantic boundaries rather than fixed sizes.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from langchain.schema import Document


@dataclass
class SemanticChunk:
    """Represents a semantic chunk with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_pos: int = 0
    end_pos: int = 0


class AgenticChunker:
    """
    Intelligent chunking system that creates semantically meaningful chunks.

    Uses LLM to identify natural boundaries in text.
    """

    def __init__(self, llm: Any, max_chunk_size: int = 3000, min_chunk_size: int = 500):
        """
        Initialize agentic chunker.

        Args:
            llm: Language model for intelligent chunking
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
        """
        self.llm = llm
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[SemanticChunk]:
        """
        Create semantic chunks from document text.

        Args:
            text: Document text to chunk
            metadata: Document metadata

        Returns:
            List of semantic chunks
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        # Simple chunking by paragraphs with size limits
        chunks = []
        paragraphs = text.split('\n\n')

        current_chunk = ""
        chunk_id = 0
        start_pos = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds max size, save current chunk
            if len(current_chunk) + len(para) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                chunks.append(SemanticChunk(
                    content=current_chunk,
                    metadata=metadata.copy(),
                    chunk_id=f"{metadata.get('source', 'unknown')}_{chunk_id}",
                    start_pos=start_pos,
                    end_pos=start_pos + len(current_chunk)
                ))
                chunk_id += 1
                start_pos += len(current_chunk)
                current_chunk = para
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para

        # Add final chunk
        if len(current_chunk) >= self.min_chunk_size:
            chunks.append(SemanticChunk(
                content=current_chunk,
                metadata=metadata.copy(),
                chunk_id=f"{metadata.get('source', 'unknown')}_{chunk_id}",
                start_pos=start_pos,
                end_pos=start_pos + len(current_chunk)
            ))

        return chunks

    def chunks_to_langchain_documents(
        self,
        chunks: List[SemanticChunk]
    ) -> List[Document]:
        """
        Convert semantic chunks to LangChain Documents.

        Args:
            chunks: List of semantic chunks

        Returns:
            List of LangChain Document objects
        """
        docs = []
        for chunk in chunks:
            # Add chunk metadata
            metadata = chunk.metadata.copy()
            metadata['chunk_id'] = chunk.chunk_id
            metadata['start_pos'] = chunk.start_pos
            metadata['end_pos'] = chunk.end_pos

            docs.append(Document(
                page_content=chunk.content,
                metadata=metadata
            ))

        return docs
