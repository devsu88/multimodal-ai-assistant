from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List
import logging
import hashlib

logger = logging.getLogger(__name__)

class ChromaStore:
    """Classe per la gestione del database vettoriale Chroma, semplificata per LangChain."""
    
    def __init__(self, persist_directory: str = "chroma_db", embedding_function: Embeddings = None):
        """
        Inizializza il database vettoriale usando il wrapper di LangChain.
        
        Args:
            persist_directory: Directory dove persistere il database.
            embedding_function: La funzione di embedding da utilizzare.
        """
        self.persist_directory = persist_directory
        self._embedding_function = embedding_function
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self._embedding_function
        )
        
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Aggiunge documenti al database, generando ID univoci.
        
        Args:
            documents: Lista di documenti da aggiungere.
            
        Returns:
            bool: True se l'aggiunta è avvenuta con successo, False altrimenti.
        """
        try:
            if not documents:
                logger.warning("Nessun documento fornito per l'aggiunta.")
                return False
                
            valid_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
            if not valid_documents:
                logger.warning("Nessun documento valido trovato dopo il filtraggio.")
                return False
            
            ids = self._generate_unique_ids(valid_documents)
            
            self.db.add_documents(documents=valid_documents, ids=ids)
            logger.info(f"Aggiunti {len(valid_documents)} chunk al database.")
            return True
            
        except Exception as e:
            logger.error(f"Errore critico durante l'aggiunta dei documenti in Chroma: {str(e)}", exc_info=True)
            return False
            
    def _generate_unique_ids(self, documents: List[Document]) -> List[str]:
        """Genera un ID univoco per ogni documento basato sul suo contenuto e metadati."""
        ids = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content
            unique_str = f"{source}-{content}"
            doc_id = hashlib.md5(unique_str.encode()).hexdigest()
            ids.append(doc_id)
        return ids

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Esegue una ricerca di similarità nel database."""
        try:
            if not query or not query.strip():
                logger.warning("Query di ricerca vuota.")
                return []
                
            return self.db.similarity_search(query, k=k)
            
        except Exception as e:
            logger.error(f"Errore durante la ricerca per similarità: {str(e)}", exc_info=True)
            return []
            
    def get_document_sources(self) -> List[str]:
        """Recupera i nomi unici dei documenti (sorgenti) dal database."""
        try:
            all_metadata = self.db.get(include=["metadatas"])
            metadatas = all_metadata.get("metadatas", [])
            
            if not metadatas:
                return []
            
            sources = {meta.get("source", "Sconosciuta") for meta in metadatas if meta}
            return sorted(list(sources))
            
        except Exception as e:
            logger.error(f"Errore nel recupero delle sorgenti dei documenti: {str(e)}", exc_info=True)
            return []

    def count(self) -> int:
        """Conta il numero di documenti nel database."""
        try:
            return self.db._collection.count()
        except Exception as e:
            logger.error(f"Errore nel conteggio dei documenti: {str(e)}", exc_info=True)
            return 0
            
    def clear(self) -> None:
        """Cancella tutti i documenti dal database."""
        try:
            self.db.delete(where={})
            logger.info("Database cancellato")
        except Exception as e:
            logger.error(f"Errore nella cancellazione del database: {str(e)}")
            raise 

    def search_by_source(self, source: str) -> List[Document]:
        """
        Cerca tutti i chunk di un documento specifico per nome file.
        
        Args:
            source: Nome del file da cercare
            
        Returns:
            Lista di Document contenenti i chunk del documento
        """
        try:
            # Cerca nel database usando il filtro sul metadata
            results = self.db.get(
                where={"source": source},
                include=["documents", "metadatas"]
            )
            
            if not results or not results['documents']:
                logger.warning(f"Nessun risultato trovato per il documento '{source}'")
                return []
                
            # Converti i risultati in Document
            documents = []
            for i, doc in enumerate(results['documents']):
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                # Verifica che il metadata.source corrisponda esattamente al source richiesto
                if metadata.get('source') == source:
                    documents.append(Document(
                        page_content=doc,
                        metadata=metadata
                    ))
                
            logger.info(f"Trovati {len(documents)} chunk per il documento '{source}'")
            return documents
            
        except Exception as e:
            logger.error(f"Errore nella ricerca per source: {str(e)}")
            return [] 