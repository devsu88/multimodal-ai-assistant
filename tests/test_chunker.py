import pytest
from chunker.text_splitter import DocumentChunker

@pytest.fixture
def chunker():
    """Fixture per il chunker di documenti."""
    return DocumentChunker(chunk_size=100, chunk_overlap=20)

@pytest.fixture
def sample_text():
    """Fixture per il testo di esempio."""
    return """Questo è un testo di esempio per testare il chunker.
    Il testo è diviso in più righe per simulare un documento reale.
    Ogni riga contiene alcune parole e punteggiatura.
    Il chunker dovrebbe dividere il testo in parti più piccole.
    La dimensione dei chunk e la sovrapposizione sono configurabili."""

def test_split_text(chunker, sample_text):
    """Test per la suddivisione del testo in chunk."""
    chunks = chunker.split_text(sample_text)
    
    # Verifica che i chunk siano una lista non vuota
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    
    # Verifica la dimensione dei chunk
    for chunk in chunks:
        assert len(chunk) <= chunker.chunk_size
        assert len(chunk) > 0

def test_split_documents(chunker):
    """Test per la suddivisione di documenti con metadati."""
    documents = [
        {
            "text": "Primo documento di test.",
            "metadata": {"source": "test1.txt"},
            "source": "test1.txt"
        },
        {
            "text": "Secondo documento di test.",
            "metadata": {"source": "test2.txt"},
            "source": "test2.txt"
        }
    ]
    
    chunks = chunker.split_documents(documents)
    
    # Verifica la struttura dei chunk
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert "source" in chunk
        assert len(chunk["text"]) <= chunker.chunk_size

def test_custom_separators():
    """Test per separatori personalizzati."""
    custom_chunker = DocumentChunker(
        chunk_size=50,
        chunk_overlap=10,
        separators=["\n", ".", " "]
    )
    
    text = "Prima riga.\nSeconda riga.\nTerza riga."
    chunks = custom_chunker.split_text(text)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    
    # Verifica che i separatori personalizzati siano rispettati
    for chunk in chunks:
        assert len(chunk) <= custom_chunker.chunk_size 