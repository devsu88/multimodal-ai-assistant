import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Classe per l'estrazione del testo da diversi tipi di documenti (PDF, immagini)."""

    def __init__(self, tesseract_path: str = None):
        """
        Inizializza l'estrattore di documenti.
        
        Args:
            tesseract_path: Path opzionale all'eseguibile di Tesseract.
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def extract(self, file_path: str) -> str:
        """
        Estrae il testo da un file, delegando al metodo corretto in base all'estensione.
        
        Args:
            file_path: Il percorso del file da cui estrarre il testo.
            
        Returns:
            Il testo estratto dal documento.
        """
        try:
            _, extension = os.path.splitext(file_path)
            extension = extension.lower()
            
            logger.info(f"Tentativo di estrazione del testo dal file: {file_path} (tipo: {extension})")

            if extension == ".pdf":
                return self._extract_from_pdf(file_path)
            elif extension in [".jpg", ".jpeg", ".png"]:
                return self._extract_from_image(file_path)
            else:
                logger.warning(f"Tipo di file non supportato: {extension}")
                return ""
        except Exception as e:
            logger.error(f"Errore durante l'estrazione del testo da {file_path}: {e}", exc_info=True)
            raise  # Rilancia l'eccezione per essere gestita a un livello superiore

    def _extract_from_pdf(self, pdf_path: str) -> str:
        """Estrae il testo da un file PDF utilizzando OCR."""
        try:
            images = convert_from_path(pdf_path)
            full_text = ""
            for i, image in enumerate(images):
                logger.info(f"Elaborazione pagina {i+1} del PDF...")
                text = pytesseract.image_to_string(image, lang='ita')
                full_text += text + "\n\n"
            logger.info("Estrazione dal PDF completata.")
            return full_text
        except Exception as e:
            logger.error(f"Errore specifico durante l'elaborazione del PDF {pdf_path}: {e}", exc_info=True)
            raise

    def _extract_from_image(self, image_path: str) -> str:
        """Estrae il testo da un file immagine, applicando pre-elaborazione per migliorare l'OCR."""
        try:
            logger.info(f"Apertura e pre-elaborazione dell'immagine: {image_path}")
            with Image.open(image_path) as img:
                # 1. Converti in scala di grigi
                img = img.convert('L')
                
                # 2. Binarizzazione (Thresholding) per creare un'immagine in bianco e nero puro
                #    Questo aiuta Tesseract a distinguere meglio il testo dallo sfondo.
                threshold = 150  # Valore sperimentale, può essere regolato
                img = img.point(lambda p: p > threshold and 255)
                
                logger.info("Pre-elaborazione completata, avvio estrazione OCR...")
                text = pytesseract.image_to_string(img, lang='ita')
                
            logger.info(f"Estrazione dall'immagine completata. Testo trovato: {len(text)} caratteri.")
            return text
        except Exception as e:
            logger.error(f"Errore specifico durante l'elaborazione dell'immagine {image_path}: {e}", exc_info=True)
            raise 