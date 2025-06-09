import pytest
import os
from vector_db.chroma_store import ChromaStore

@pytest.fixture
def vector_store():
    """Fixture per il database vettoriale."""
    # Usa una directory temporanea per i test
    test_dir = "./test_chroma_db"
    store = ChromaStore(persist_directory=test_dir)
    yield store
    # Pulisci dopo i test
    store.clear()
    if os.path.exists(test_dir):
        os.rmdir(test_dir)

@pytest.fixture
def sample_documents():
    """Fixture per documenti di esempio."""
    return [
        {
            "text": "Questo è il primo documento di test.",
            "metadata": {"source": "test1.txt"}
        },
        {
            "text": "Questo è il secondo documento di test.",
            "metadata": {"source": "test2.txt"}
        }
    ]

def test_add_documents(vector_store, sample_documents):
    """Test per l'aggiunta di documenti al database."""
    vector_store.add_documents(sample_documents)
    
    # Verifica la ricerca
    results = vector_store.search("documento di test")
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Verifica la struttura dei risultati
    for result in results:
        assert "text" in result
        assert "metadata" in result
        assert "distance" in result

def test_search_relevance(vector_store, sample_documents):
    """Test per la rilevanza dei risultati di ricerca."""
    vector_store.add_documents(sample_documents)
    
    # Cerca con query specifica
    results = vector_store.search("primo documento")
    
    # Verifica che il primo documento sia più rilevante
    assert len(results) > 0
    assert "primo" in results[0]["text"].lower()

def test_clear_database(vector_store, sample_documents):
    """Test per la cancellazione del database."""
    vector_store.add_documents(sample_documents)
    vector_store.clear()
    
    # Verifica che la ricerca non restituisca risultati
    results = vector_store.search("test")
    assert len(results) == 0

def test_custom_embedding_model():
    """Test per l'uso di un modello di embedding personalizzato."""
    store = ChromaStore(embedding_model="all-MiniLM-L6-v2")
    assert store.embedding_model is not None 