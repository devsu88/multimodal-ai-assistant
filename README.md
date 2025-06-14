# 🤖 Assistente Documenti Multimodale

Un assistente AI avanzato in grado di elaborare e analizzare documenti in vari formati, fornendo risposte intelligenti alle domande degli utenti.

## 🚀 Caratteristiche Principali

- **Elaborazione Multimodale**: Supporto per documenti PDF, TXT e DOCX
- **OCR Integrato**: Estrazione del testo da immagini e documenti scansionati
- **Memoria Conversazionale**: Mantiene il contesto delle conversazioni
- **Interfaccia Web**: UI intuitiva basata su Streamlit
- **Base di Conoscenza Vettoriale**: Archiviazione efficiente e ricerca semantica dei documenti

## 📋 Prerequisiti

- Python 3.8+
- Tesseract OCR
- Ollama (per il modello LLM locale)

## 🛠️ Installazione

1. Clona il repository:
```bash
git clone https://github.com/tuousername/multimodal-ai-assistant.git
cd multimodal-ai-assistant
```

2. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

3. Installa Tesseract OCR:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Scarica l'installer da https://github.com/UB-Mannheim/tesseract/wiki
```

4. Installa e avvia Ollama:
```bash
# Segui le istruzioni su https://ollama.ai
```

## 🚀 Avvio

1. Avvia l'applicazione:
```bash
streamlit run app.py
```

2. Apri il browser all'indirizzo indicato (solitamente http://localhost:8501)

## 📁 Struttura del Progetto

```
multimodal-ai-assistant/
├── agent/                 # Logica dell'agente conversazionale
├── chunker/              # Elaborazione e suddivisione dei documenti
├── ocr/                  # Funzionalità OCR
├── text_processing/      # Elaborazione del testo
├── ui/                   # Componenti dell'interfaccia utente
├── vector_db/           # Gestione del database vettoriale
├── tests/               # Test unitari e di integrazione
├── app.py               # Punto di ingresso dell'applicazione
└── requirements.txt     # Dipendenze del progetto
```

## 💡 Utilizzo

1. Carica un documento utilizzando l'interfaccia web
2. Attendi l'elaborazione del documento
3. Inizia a fare domande sul contenuto del documento
4. L'assistente fornirà risposte basate sul contesto del documento

## 🧪 Test

Esegui i test unitari:
```bash
pytest tests/
```

## 🤝 Contribuire

Le contribuzioni sono benvenute! Per favore:

1. Fai un fork del repository
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Committa le tue modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Pusha al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## 📝 Licenza

Questo progetto è distribuito con licenza MIT. Vedi il file `LICENSE` per maggiori dettagli.

## 🙏 Ringraziamenti

- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Streamlit](https://streamlit.io)
- [Ollama](https://ollama.ai) 