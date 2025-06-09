import re
from typing import List, Dict, Any, Optional
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunker:
    """Classe per la suddivisione del testo in chunk."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        """
        Inizializza il chunker.
        
        Args:
            chunk_size: Dimensione massima di ogni chunk
            chunk_overlap: Sovrapposizione tra chunk consecutivi
            min_chunk_size: Dimensione minima di un chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        logger.info(f"Chunker inizializzato con chunk_size={chunk_size}, overlap={chunk_overlap}")
        
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Suddivide il testo in chunk.
        
        Args:
            text: Testo da suddividere
            metadata: Metadati da associare ai chunk (opzionale)
            
        Returns:
            Lista di documenti (chunk)
        """
        # Se non vengono forniti metadati, usa un dizionario vuoto
        if metadata is None:
            metadata = {}
            
        # Crea i documenti con i metadati associati
        # create_documents si aspetta una lista di metadati, una per ogni documento
        # Dato che abbiamo un solo testo, passiamo una lista con un solo elemento
        chunks = self.text_splitter.create_documents([text], metadatas=[metadata])
        
        return chunks
        
    def _clean_text(self, text: str) -> str:
        """
        Pulisce il testo da caratteri indesiderati.
        
        Args:
            text: Testo da pulire
            
        Returns:
            Testo pulito
        """
        # Rimuovi spazi multipli
        text = re.sub(r'\s+', ' ', text)
        # Rimuovi caratteri di controllo
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        return text.strip()
        
    def _split_paragraphs(self, text: str) -> List[str]:
        """
        Suddivide il testo in paragrafi.
        
        Args:
            text: Testo da suddividere
            
        Returns:
            Lista di paragrafi
        """
        # Suddividi su newline o punti
        paragraphs = re.split(r'\n+|(?<=\.)\s+', text)
        # Filtra paragrafi vuoti o troppo corti
        return [p.strip() for p in paragraphs if len(p.strip()) >= self.min_chunk_size]
        
    def _split_large_paragraph(self, paragraph: str) -> List[Dict[str, Any]]:
        """
        Suddivide un paragrafo grande in chunk più piccoli.
        
        Args:
            paragraph: Paragrafo da suddividere
            
        Returns:
            Lista di chunk
        """
        chunks = []
        words = paragraph.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 per lo spazio
            
            if current_size + word_size > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk([' '.join(current_chunk)]))
                    # Mantieni l'overlap
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_size = sum(len(w) + 1 for w in current_chunk)
            
            current_chunk.append(word)
            current_size += word_size
        
        if current_chunk:
            chunks.append(self._create_chunk([' '.join(current_chunk)]))
        
        return chunks
        
    def _create_chunk(self, paragraphs: List[str]) -> Dict[str, Any]:
        """
        Crea un chunk con metadati.
        
        Args:
            paragraphs: Lista di paragrafi
            
        Returns:
            Chunk con metadati
        """
        text = ' '.join(paragraphs)
        return {
            "text": text,
            "metadata": {
                "chunk_size": len(text),
                "num_paragraphs": len(paragraphs)
            }
        } 