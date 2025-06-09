# Assistente Documentale Multimodale

Un assistente AI in grado di elaborare documenti PDF e immagini, estrarre il testo tramite OCR, e rispondere a domande sul contenuto utilizzando tecniche di RAG (Retrieval Augmented Generation).

## Caratteristiche

- 🖼️ Supporto per PDF e immagini (JPEG/PNG)
- 📝 Estrazione testo con Tesseract OCR
- 🔍 Chunking del testo con LangChain
- 🧮 Embeddings con sentence-transformers
- 💾 Archiviazione vettoriale con ChromaDB
- 🤖 Generazione risposte con Ollama
- 💬 Interfaccia web con Streamlit

## Requisiti

- Python 3.11+
- Tesseract OCR
- Ollama (per il modello LLM locale)

## Installazione

1. Clona il repository:
```bash
git clone https://github.com/yourusername/multimodal-ai-assistant.git
cd multimodal-ai-assistant
```

2. Crea un ambiente virtuale:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
.\venv\Scripts\activate  # Windows
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

4. Installa Tesseract OCR:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-ita  # per il supporto italiano

# macOS
brew install tesseract
brew install tesseract-lang  # per le lingue aggiuntive

# Windows
# Scarica l'installer da https://github.com/UB-Mannheim/tesseract/wiki
```

5. Installa Ollama:
```bash
# Segui le istruzioni su https://ollama.ai/download
```

## Configurazione

1. Copia il file di esempio delle variabili d'ambiente:
```bash
cp .env.example .env
```

2. Modifica il file `.env` con le tue configurazioni:
```
TESSERACT_CMD=/usr/bin/tesseract  # percorso al comando tesseract
OLLAMA_MODEL=llama2  # modello Ollama da utilizzare
```

## Utilizzo

1. Avvia l'applicazione:
```bash
streamlit run ui/app.py
```

2. Apri il browser all'indirizzo indicato (default: http://localhost:8501)

3. Carica i documenti tramite l'interfaccia web

4. Inizia a fare domande sul contenuto dei documenti

## Test

Esegui i test unitari:
```bash
pytest tests/
```

## Struttura del Progetto

```
multimodal-ai-assistant/
├── ocr/
│   └── extractor.py
├── chunker/
│   └── text_splitter.py
├── vector_db/
│   └── chroma_store.py
├── agent/
│   └── conversation_agent.py
├── ui/
│   └── app.py
├── tests/
│   ├── test_ocr.py
│   ├── test_chunker.py
│   └── test_vector_db.py
├── requirements.txt
├── .env.example
└── README.md
```

## Contribuire

1. Fork il repository
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Commit le tue modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## Licenza

Distribuito sotto la licenza MIT. Vedi `LICENSE` per maggiori informazioni.

## Contatti

Salvatore Ucchino - developer.su120188@gmail.com

Link al Progetto: [https://github.com/devsu88/multimodal-ai-assistant](https://github.com/yourusername/multimodal-ai-assistant) 