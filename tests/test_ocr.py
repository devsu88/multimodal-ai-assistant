import pytest
import os
from ocr.extractor import DocumentExtractor

@pytest.fixture
def extractor():
    """Fixture per l'estrattore di documenti."""
    return DocumentExtractor()

@pytest.fixture
def test_image_path():
    """Fixture per il percorso dell'immagine di test."""
    return "tests/data/test_image.png"

@pytest.fixture
def test_pdf_path():
    """Fixture per il percorso del PDF di test."""
    return "tests/data/test_document.pdf"

def test_extract_from_image(extractor, test_image_path):
    """Test per l'estrazione del testo da un'immagine."""
    if not os.path.exists(test_image_path):
        pytest.skip(f"File di test non trovato: {test_image_path}")
        
    text = extractor.extract_from_image(test_image_path)
    assert isinstance(text, str)
    assert len(text) > 0

def test_extract_from_pdf(extractor, test_pdf_path):
    """Test per l'estrazione del testo da un PDF."""
    if not os.path.exists(test_pdf_path):
        pytest.skip(f"File di test non trovato: {test_pdf_path}")
        
    text = extractor.extract_from_pdf(test_pdf_path)
    assert isinstance(text, str)
    assert len(text) > 0
    assert "--- Pagina" in text  # Verifica la formattazione delle pagine

def test_extract_invalid_file(extractor):
    """Test per il gestione di file non validi."""
    with pytest.raises(FileNotFoundError):
        extractor.extract("file_non_esistente.pdf")
        
    with pytest.raises(ValueError):
        extractor.extract("file.txt")  # Formato non supportato 