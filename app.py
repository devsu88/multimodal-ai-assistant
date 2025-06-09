import streamlit as st
from agent.conversation_agent import ConversationAgent
from utils.llm_utils import get_llm, get_embeddings
from utils.vector_store import get_vector_store

# Configurazione della pagina
st.set_page_config(
    page_title="Assistente Documenti",
    page_icon="📚",
    layout="wide"
)

# Titolo dell'applicazione
st.title("🤖 Assistente Documenti")

# Inizializza l'agente di conversazione
if "conversation_agent" not in st.session_state:
    llm = get_llm()
    embeddings = get_embeddings()
    vector_store = get_vector_store()
    st.session_state.conversation_agent = ConversationAgent(llm, embeddings, vector_store)

# Inizializza la chat con il messaggio di benvenuto
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": st.session_state.conversation_agent.get_welcome_message()
        }
    ]

# Area di caricamento file
uploaded_file = st.file_uploader("Carica un documento", type=["pdf", "txt", "docx"])

if uploaded_file is not None:
    # Salva il file temporaneamente
    with open(f"temp_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Aggiungi il documento al vector store
    st.session_state.conversation_agent.add_document(f"temp_{uploaded_file.name}")
    
    st.success(f"Documento {uploaded_file.name} caricato con successo!")

# Mostra la cronologia dei messaggi
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dell'utente
if prompt := st.chat_input("Fai una domanda sui documenti"):
    # Aggiungi il messaggio dell'utente alla cronologia
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Mostra il messaggio dell'utente
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Genera la risposta
    with st.chat_message("assistant"):
        response = st.session_state.conversation_agent.process_query(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response}) 