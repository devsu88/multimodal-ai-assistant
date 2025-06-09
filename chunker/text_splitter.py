from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunker:
    """Classe per la suddivisione del testo in chunk."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        Inizializza il chunker di documenti.
        
        Args:
            chunk_size: Dimensione massima di ogni chunk
            chunk_overlap: Sovrapposizione tra chunk consecutivi
            separators: Lista di separatori per la suddivisione del testo
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )
        
    def split_text(self, text: str) -> List[str]:
        """
        Suddivide il testo in chunk.
        
        Args:
            text: Testo da suddividere
            
        Returns:
            Lista di chunk di testo
        """
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Testo suddiviso in {len(chunks)} chunk")
            return chunks
        except Exception as e:
            logger.error(f"Errore nella suddivisione del testo: {str(e)}")
            raise
            
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Suddivide una lista di documenti in chunk.
        
        Args:
            documents: Lista di documenti con metadati
            
        Returns:
            Lista di chunk con metadati
        """
        try:
            chunks = []
            for doc in documents:
                doc_chunks = self.text_splitter.split_text(doc["text"])
                for chunk in doc_chunks:
                    chunks.append({
                        "text": chunk,
                        "metadata": doc.get("metadata", {}),
                        "source": doc.get("source", "unknown")
                    })
            logger.info(f"Documenti suddivisi in {len(chunks)} chunk totali")
            return chunks
        except Exception as e:
            logger.error(f"Errore nella suddivisione dei documenti: {str(e)}")
            raise 