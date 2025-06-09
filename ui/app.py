import streamlit as st
import os
from typing import List, Dict, Any
import tempfile
from datetime import datetime
import sys
from pathlib import Path
from langchain_community.chat_models import ChatOllama
from langchain.tools import Tool
import logging
import hashlib
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import LLMChain
from langchain_core.messages import AIMessage, HumanMessage

# Aggiungi la directory principale al path
sys.path.append(str(Path(__file__).parent.parent))

from ocr.extractor import DocumentExtractor
from text_processing.chunker import DocumentChunker
from vector_db.chroma_store import ChromaStore
from agent.conversation_agent import ConversationAgent

# Configurazione della pagina
st.set_page_config(
    page_title="Assistente Documentale Multimodale",
    page_icon="📚",
    layout="wide"
)

# Configurazione del logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_components() -> Dict[str, Any]:
    """Inizializza e restituisce i componenti principali dell'applicazione come dizionario."""
    try:
        embedding_model = OllamaEmbeddings(model="qwen2:7b", base_url="http://localhost:11434")
        llm = ChatOllama(model="qwen2:7b", temperature=0.1)
        vector_store = ChromaStore(
            persist_directory="chroma_db",
            embedding_function=embedding_model
        )
        chunker = DocumentChunker()
        extractor = DocumentExtractor()
        agent = ConversationAgent(llm=llm, vector_store=vector_store)
        
        return {
            "vector_store": vector_store,
            "chunker": chunker,
            "extractor": extractor,
            "agent": agent
        }
    except Exception as e:
        logger.error(f"Errore nell'inizializzazione dei componenti: {str(e)}", exc_info=True)
        st.error("Si è verificato un errore nell'inizializzazione dell'applicazione. Controlla i log per maggiori dettagli.")
        raise

def initialize_app_state():
    """Inizializza lo stato della sessione per componenti e messaggi."""
    if "components" not in st.session_state:
        logger.info("Inizializzazione componenti per la prima volta...")
        st.session_state.components = init_components()
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": st.session_state.components["agent"].get_welcome_message()
            }
        ]
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

def handle_file_upload():
    """Gestisce il caricamento dei documenti."""
    components = st.session_state.components
    uploaded_files = st.file_uploader(
        "Carica i tuoi documenti",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        return
        
    for file in uploaded_files:
        if not file:
            continue
            
        file_hash = hashlib.md5(file.getvalue()).hexdigest()
        if file_hash in st.session_state.processed_files:
            if len(uploaded_files) > len(st.session_state.processed_files):
                continue
            st.info(f"Il documento {file.name} è già stato elaborato.")
            continue
        
        temp_path = f"temp_{file.name}"
        try:
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            
            with st.spinner(f"Elaborazione di {file.name}..."):
                text = components["extractor"].extract(temp_path)
                
                if not text:
                    st.warning(f"Il documento {file.name} non contiene testo estraibile.")
                    continue
                
                chunks = components["chunker"].chunk_text(
                    text, 
                    metadata={"source": file.name, "hash": file_hash}
                )
                
                if not chunks:
                    st.warning(f"Impossibile suddividere il documento {file.name} in chunk.")
                    continue
                
                success = components["vector_store"].add_documents(chunks)
                
                if success:
                    st.session_state.processed_files.add(file_hash)
                    st.success(f"Documento {file.name} elaborato con successo!")
                    st.rerun()
                else:
                    st.error(f"Errore nel salvataggio del documento {file.name} nel database.")
            
        except Exception as e:
            logger.error(f"Errore nell'elaborazione di {file.name}: {str(e)}", exc_info=True)
            st.error(f"Errore nell'elaborazione di {file.name}: {str(e)}")
        
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Impossibile rimuovere il file temporaneo {temp_path}: {str(e)}")

def handle_chat(prompt: str):
    """Gestisce l'interazione chat con l'utente."""
    if not prompt or not prompt.strip():
        return
        
    components = st.session_state.components
    try:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Elaborazione della risposta..."):
            response = components["agent"].process_query(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
        
    except Exception as e:
        logger.error(f"Errore nella gestione della chat: {str(e)}", exc_info=True)
        st.error(f"Si è verificato un errore: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Mi dispiace, si è verificato un errore nell'elaborazione della tua richiesta."
        })

def main():
    """Funzione principale dell'applicazione."""
    st.title("📚 Assistente Documentale Multimodale")
    
    initialize_app_state()
    components = st.session_state.components
    
    doc_count = components["vector_store"].count()
    st.sidebar.info(f"Chunk indicizzati: {doc_count}")
    
    with st.sidebar:
        st.header("Carica Documenti")
        handle_file_upload()
    
    st.header("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Fai una domanda sui tuoi documenti..."):
        handle_chat(prompt)

if __name__ == "__main__":
    main() 