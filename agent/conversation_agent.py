import os
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Optional, Union
import logging
import json
from langchain_core.language_models import BaseChatModel
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from vector_db.chroma_store import ChromaStore

from langchain.tools import Tool, StructuredTool
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants, Schemas, and Helper Texts ---

AGENT_PROMPT_TEMPLATE = """Sei un assistente AI specializzato nell'analisi di documenti. La tua risposta DEVE essere unicamente un blocco di codice JSON e nient'altro.

Hai accesso ai seguenti strumenti:
{tools}

I valori validi per il campo "action" nel JSON sono "Final Answer" o uno tra: {tool_names}

---
ESEMPIO 1:
Domanda: "c'è qualcosa sull'intelligenza artificiale?"
```json
{{
  "action": "search_document_content",
  "action_input": {{ "query": "intelligenza artificiale" }}
}}
```
---
ESEMPIO 2:
Domanda: "leggimi il file report_annuale.pdf" o "cosa contiene report_annuale.pdf"
```json
{{
  "action": "extract_full_document_text",
  "action_input": {{ "document_name": "report_annuale.pdf" }}
}}
```
---
ESEMPIO 3:
Domanda: "i documenti A e B sono correlati?" o "ci sono collegamenti tra A e B?"
```json
{{
  "action": "analyze_document_correlation",
  "action_input": {{ 
    "query": "argomento specifico",  // opzionale
    "min_correlation": 0.3  // opzionale, default 0.3
  }}
}}
```
---
ESEMPIO 4:
Domanda: "ciao"
```json
{{
  "action": "Final Answer",
  "action_input": "Ciao! Sono pronto ad aiutarti con i tuoi documenti. Cosa vuoi sapere?"
}}
```
---

REGOLE IMPORTANTI:
1. Usa "extract_full_document_text" quando l'utente chiede il contenuto di un documento specifico
2. Usa "search_document_content" solo per cercare informazioni specifiche in tutti i documenti
3. Usa "analyze_document_correlation" quando l'utente chiede relazioni tra documenti
   - Puoi specificare una query opzionale per filtrare i risultati
   - Puoi specificare una soglia di correlazione (default 0.3)
4. Dopo aver ottenuto i risultati, restituisci sempre una "Final Answer" con il contenuto formattato
   - IMPORTANTE: per "Final Answer", action_input DEVE essere una stringa, non un dizionario
   - Esempio corretto: "action_input": "Ecco il contenuto del documento..."
   - Esempio errato: "action_input": {{ "content": "Ecco il contenuto..." }}
5. Non entrare in loop di ricerca, usa i risultati trovati per formare la risposta finale
6. Per domande sulla correlazione tra documenti, usa sempre lo strumento specifico invece di dare risposte generiche

Ora, rispondi alla domanda dell'utente.

Cronologia conversazione:
{chat_history}

Domanda utente: {input}

Passaggi intermedi (le tue riflessioni e azioni passate):
{agent_scratchpad}
"""

SYSTEM_COMMANDS = {"help", "info", "aiuto", "come funziona"}

HELP_MESSAGES = {
    "welcome": """👋 Benvenuto! Sono il tuo assistente AI specializzato nell'analisi dei documenti.

Posso aiutarti a:
- 🔍 Cercare informazioni specifiche nei tuoi documenti
- 📄 Estrarre e leggere il contenuto completo dei documenti
- 💡 Rispondere a domande dettagliate sul contenuto
- 📋 Fornire un elenco dei documenti disponibili
- 📝 Generare riassunti dei documenti

Per iniziare:
1. Carica almeno un documento usando il pulsante "Carica un documento"
2. Fammi una domanda sul contenuto
3. Ti aiuterò a trovare le informazioni che cerchi!

Sono qui per rendere più semplice e veloce l'analisi dei tuoi documenti. Come posso aiutarti oggi? 😊""",
    "general": """Ciao! Sono il tuo assistente per l'analisi dei documenti. Posso aiutarti a:
- Cercare informazioni nei documenti
- Estrarre il testo dai documenti
- Elencare i documenti disponibili
- Rispondere a domande sul contenuto dei documenti

Per iniziare, carica almeno un documento e fammi una domanda!""",
    "system": """Sistema di Analisi Documenti
- Supporta documenti PDF e immagini
- Utilizza OCR per estrarre testo dalle immagini
- Permette ricerche semantiche nel contenuto
- Mantiene la cronologia delle conversazioni

Per iniziare, carica un documento e fammi una domanda!""",
    "usage": """Ecco come utilizzare l'assistente:

1. Carica un documento:
   - Supporta file PDF e immagini
   - Puoi caricare più documenti

2. Fai domande come:
   - "Quali documenti sono presenti?"
   - "Cosa contiene il file X?"
   - "Cerca informazioni su Y"
   - "Estrai il testo dal documento Z"

3. Comandi di sistema:
   - "Aiuto" o "Help" per queste istruzioni
   - "Info" per informazioni sul sistema
   - "Come funziona?" per istruzioni d'uso

Ricorda: devi caricare almeno un documento prima di poter fare domande sul contenuto!"""
}

class EmptyInput(BaseModel):
    pass

class DocumentSearchInput(BaseModel):
    query: str = Field(description="La domanda o l'argomento specifico da cercare nei documenti.")

class DocumentExtractInput(BaseModel):
    document_name: str = Field(description="Il nome del file (o una sua parte univoca) da cui estrarre il testo.")

class DocumentCorrelationInput(BaseModel):
    query: Optional[str] = Field(default=None, description="Query opzionale per filtrare le correlazioni su un tema specifico")
    min_correlation: float = Field(default=0.3, description="Soglia minima di correlazione (0-1)")

class CustomOutputParser(StructuredChatOutputParser):
    """Parser personalizzato per gestire risposte in testo libero come Final Answer."""
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            # Prova a fare il parsing come JSON strutturato
            return super().parse(text)
        except Exception as e:
            logger.warning(f"Parsing fallito: {str(e)}. Avvolgimento della risposta in Final Answer.")
            # Se il parsing fallisce, assumi che sia una risposta finale
            return AgentFinish(return_values={"output": text}, log=text)

class ConversationAgent:
    """Agente conversazionale avanzato basato su strumenti."""
    
    def __init__(self, llm: BaseChatModel, vector_store: ChromaStore):
        self.llm = llm
        self.vector_store = vector_store
        
        tools = self._setup_tools()
        prompt = self._setup_prompt()
        
        # Creiamo l'agente strutturato con il parser personalizzato
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True,
            memory=self.memory,
            max_iterations=10,  # Ridotto a 5 per evitare loop
            early_stopping_method="force",
            return_intermediate_steps=True  # Aggiunto per debug
        )

    def _setup_tools(self) -> List[Tool]:
        """Definisce e configura gli strumenti che l'agente può utilizzare."""
        return [
            StructuredTool(
                name="list_available_documents",
                func=self.list_documents,
                description="Elenca tutti i documenti attualmente disponibili. Usalo quando l'utente chiede quali file ci sono o quanti documenti sono caricati.",
                args_schema=EmptyInput,
                handle_tool_error=True
            ),
            Tool(
                name="search_document_content",
                func=self.search_documents,
                description="Cerca un'informazione specifica all'interno del contenuto di tutti i documenti disponibili. Usalo per rispondere a domande su argomenti specifici.",
                args_schema=DocumentSearchInput,
                handle_tool_error=True
            ),
            Tool(
                name="extract_full_document_text",
                func=self.extract_document_text,
                description="Estrae e restituisce il testo completo di un singolo documento specifico. Usalo quando l'utente chiede esplicitamente di leggere o vedere il contenuto di un file.",
                args_schema=DocumentExtractInput,
                handle_tool_error=True
            ),
            StructuredTool(
                name="analyze_document_correlation",
                func=self.analyze_document_correlation,
                description="Analizza la correlazione tra due documenti specifici. Usalo quando l'utente chiede se ci sono collegamenti o relazioni tra documenti.",
                args_schema=DocumentCorrelationInput,
                handle_tool_error=True
            )
        ]

    def _setup_prompt(self) -> ChatPromptTemplate:
        """Crea il prompt per l'agente usando un template predefinito."""
        return ChatPromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)

    def process_query(self, query: str) -> str:
        """Processa la query dell'utente e restituisce una risposta."""
        logger.info(f"Inizio elaborazione query: {query}")
        normalized_query = query.lower().strip()

        # Gestione dei casi preliminari
        if not self.has_documents():
            # Se non ci sono documenti, l'agente può solo dare il benvenuto o rispondere a comandi di sistema
            if normalized_query in SYSTEM_COMMANDS:
                 return self._handle_system_command(normalized_query)
            return self._get_welcome_message()
        
        # Gestione query di sistema anche quando ci sono documenti
        if normalized_query in SYSTEM_COMMANDS:
            return self._handle_system_command(normalized_query)

        try:
            logger.info("Esecuzione dell'agente principale.")
            response = self.agent_executor.invoke({
                "input": query,
                "chat_history": self.memory.chat_memory.messages if self.memory else []
            })
            
            # Gestione della risposta dell'agente
            output = response.get("output", "Non ho trovato una risposta.")
            
            # Se l'output è un dizionario con action_input, estraiamo il contenuto
            if isinstance(output, dict):
                if "action_input" in output:
                    action_input = output["action_input"]
                    if isinstance(action_input, dict):
                        if "content" in action_input:
                            output = action_input["content"]
                        else:
                            output = str(action_input)
                    else:
                        output = str(action_input)
                elif "content" in output:
                    output = output["content"]
                else:
                    output = str(output)
            
            # Se non è una stringa, la convertiamo
            if not isinstance(output, str):
                output = str(output)
                
            # Puliamo la risposta
            cleaned_output = self._clean_response(output)
            
            # Se la risposta è il messaggio di benvenuto generico, proviamo a usare l'ultima azione dell'agente
            if cleaned_output == "Ciao! Sono pronto ad aiutarti con i tuoi documenti. Cosa vuoi sapere?":
                intermediate_steps = response.get("intermediate_steps", [])
                if intermediate_steps:
                    last_step = intermediate_steps[-1]
                    if isinstance(last_step, tuple) and len(last_step) > 1:
                        last_output = last_step[1]
                        if isinstance(last_output, str):
                            cleaned_output = last_output
            
            return cleaned_output
        
        except Exception as e:
            logger.error(f"Errore critico durante l'esecuzione dell'agente: {e}", exc_info=True)
            return self._handle_direct_query(query)

    def _clean_response(self, response: str) -> str:
        """Pulisce la risposta da formattazioni non necessarie."""
        if not response:
            return "Mi dispiace, non sono riuscito a generare una risposta."
        
        # Rimuovi formattazioni non necessarie come i backticks del codice
        response = response.strip()
        if response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        
        # Assicuriamoci che la risposta sia una stringa valida
        if isinstance(response, dict):
            if "content" in response:
                response = response["content"]
            else:
                response = str(response)
            
        return response

    def has_documents(self) -> bool:
        """Verifica se ci sono documenti nel database."""
        try:
            return self.vector_store.count() > 0
        except Exception as e:
            logger.error(f"Errore nel conteggio dei documenti: {str(e)}")
            return False

    def _handle_system_command(self, command: str) -> str:
        """Gestisce le query di sistema in base a parole chiave."""
        command = command.lower().strip()
        if command in ["help", "aiuto", "come funziona"]:
            return HELP_MESSAGES["usage"]
        if command == "info":
            return HELP_MESSAGES["system"]
        return HELP_MESSAGES["general"]

    def list_documents(self) -> str:
        """Elenca i documenti presenti nel database."""
        try:
            logger.info("Richiesta di elenco documenti")
            if not self.has_documents():
                logger.info("Nessun documento presente nel sistema")
                return "Non ci sono documenti caricati nel sistema."

            sources = self.vector_store.get_document_sources()
            num_docs = len(sources)
            logger.info(f"Trovati {num_docs} documenti nel sistema")

            if num_docs == 1:
                logger.info(f"Documento trovato: {sources[0]}")
                return f"Attualmente c'è 1 documento nel sistema: {sources[0]}"
            
            doc_list = "\n - ".join(sources)
            logger.info(f"Documenti trovati: {doc_list}")
            return f"Attualmente ci sono {num_docs} documenti nel sistema:\n - {doc_list}"
            
        except Exception as e:
            logger.error(f"Errore nell'elencazione dei documenti: {str(e)}")
            return "Si è verificato un errore nel recupero della lista dei documenti."

    def search_documents(self, query: str) -> str:
        """Cerca informazioni nei documenti."""
        try:
            logger.info(f"Richiesta di ricerca con query: {query}")
            if not self.has_documents():
                logger.warning("Tentativo di ricerca senza documenti disponibili")
                return "Non ci sono documenti disponibili per la ricerca."

            results = self.vector_store.search(query)
            logger.info(f"Trovati {len(results)} risultati per la query")
            
            if not results:
                logger.info("Nessun risultato trovato per la query")
                return "Non ho trovato informazioni pertinenti nei documenti per la tua domanda."

            formatted_results = []
            for doc in results:
                source = doc.metadata.get('source', 'Sconosciuta')
                content = doc.page_content.strip()
                logger.debug(f"Risultato trovato nel documento '{source}'")
                formatted_results.append(f"Il documento '{source}' contiene:\n{content}")
            
            return "\n\n---\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Errore nella ricerca dei documenti: {str(e)}")
            return "Si è verificato un errore durante la ricerca nei documenti."

    def extract_document_text(self, document_name: str) -> str:
        """Estrae il testo da un documento specifico."""
        try:
            logger.info(f"Richiesta di estrazione testo per il documento: {document_name}")
            if not self.has_documents():
                logger.warning("Tentativo di estrazione testo senza documenti disponibili")
                return "Non ci sono documenti disponibili per l'estrazione del testo."

            target_source = document_name

            # Cerca il documento nel database
            sources = self.vector_store.get_document_sources()
            matching_sources = [s for s in sources if target_source.lower() in s.lower()]
            
            if not matching_sources:
                logger.warning(f"Nessun documento trovato che contenga '{target_source}'")
                return f"Nessun documento trovato che contenga '{target_source}' nel nome. Documenti disponibili:\n" + "\n".join(f"- {s}" for s in sources)
            
            if len(matching_sources) > 1:
                logger.warning(f"Trovati {len(matching_sources)} documenti che corrispondono a '{target_source}'")
                return f"Ho trovato più documenti che corrispondono a '{target_source}':\n" + "\n".join(f"- {s}" for s in matching_sources) + "\n\nPuoi specificare meglio quale documento vuoi analizzare?"
            
            # Estrai il testo dal documento trovato
            logger.info(f"Estrazione testo dal documento: {matching_sources[0]}")
            results = self.vector_store.search_by_source(matching_sources[0])
            
            if not results:
                logger.warning(f"Impossibile estrarre il testo dal documento '{matching_sources[0]}'")
                return f"Non sono riuscito a estrarre il testo dal documento '{matching_sources[0]}'."

            # Estrai e formatta il testo
            text_parts = []
            for doc in results:
                text_parts.append(doc.page_content.strip())
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Testo estratto con successo dal documento '{matching_sources[0]}'")
            return f"Ecco il contenuto del documento '{matching_sources[0]}':\n\n{full_text}"
            
        except Exception as e:
            logger.error(f"Errore nell'estrazione del testo: {str(e)}")
            return "Si è verificato un errore durante l'estrazione del testo dal documento."

    def analyze_document_correlation(self, query: Optional[str] = None, min_correlation: float = 0.3) -> str:
        """Analizza le correlazioni tra tutti i documenti disponibili, opzionalmente filtrate per una query specifica."""
        try:
            logger.info(f"Analisi correlazioni tra documenti" + (f" per query: {query}" if query else ""))
            
            # Ottieni tutti i documenti disponibili
            sources = self.vector_store.get_document_sources()
            if not sources:
                return "Non ci sono documenti disponibili per l'analisi delle correlazioni."
            
            # Se c'è una query, filtra prima i documenti
            if query:
                results = self.vector_store.search(query)
                if not results:
                    return f"Non ho trovato informazioni pertinenti alla query '{query}' nei documenti."
                doc_results = {}
                for doc in results:
                    source = doc.metadata.get('source', 'Sconosciuta')
                    if source not in doc_results:
                        doc_results[source] = []
                    doc_results[source].append(doc.page_content)
            else:
                # Se non c'è query, analizza tutti i documenti
                doc_results = {}
                for source in sources:
                    results = self.vector_store.search_by_source(source)
                    if results:
                        doc_results[source] = [doc.page_content for doc in results]
            
            if len(doc_results) < 2:
                return "Sono necessari almeno due documenti per analizzare le correlazioni."
            
            # Analizza le correlazioni
            correlations = []
            
            # 1. Analisi delle parole chiave comuni
            doc_keywords = {}
            for source, contents in doc_results.items():
                # Estrai parole chiave significative
                keywords = set()
                for content in contents:
                    # Dividi in frasi per un'analisi più precisa
                    sentences = content.split('.')
                    for sentence in sentences:
                        words = sentence.lower().split()
                        # Filtra parole corte e comuni
                        keywords.update([w for w in words if len(w) > 3 and w not in {'quale', 'quali', 'quando', 'dove', 'come', 'perché'}])
                doc_keywords[source] = keywords
            
            # 2. Calcola la similarità tra documenti
            doc_pairs = []
            for i, (source1, keywords1) in enumerate(doc_keywords.items()):
                for source2, keywords2 in list(doc_keywords.items())[i+1:]:
                    # Calcola similarità Jaccard
                    intersection = len(keywords1.intersection(keywords2))
                    union = len(keywords1.union(keywords2))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity >= min_correlation:
                        doc_pairs.append((source1, source2, similarity))
            
            # 3. Formatta i risultati
            if doc_pairs:
                if query:
                    correlations.append(f"Ho trovato le seguenti correlazioni tra i documenti relativi a '{query}':")
                else:
                    correlations.append("Ho trovato le seguenti correlazioni tra i documenti:")
                
                # Raggruppa i documenti per livello di correlazione
                high_corr = []  # > 0.7
                medium_corr = []  # 0.4-0.7
                low_corr = []  # 0.3-0.4
                
                for source1, source2, similarity in sorted(doc_pairs, key=lambda x: x[2], reverse=True):
                    similarity_percent = int(similarity * 100)
                    pair_info = f"- '{source1}' e '{source2}' sono correlati al {similarity_percent}%"
                    
                    # Aggiungi dettagli sulle parole chiave comuni
                    common_keywords = doc_keywords[source1].intersection(doc_keywords[source2])
                    if common_keywords:
                        pair_info += f"\n  Parole chiave comuni: {', '.join(list(common_keywords)[:5])}"
                    
                    if similarity > 0.7:
                        high_corr.append(pair_info)
                    elif similarity > 0.4:
                        medium_corr.append(pair_info)
                    else:
                        low_corr.append(pair_info)
                
                # Aggiungi i risultati raggruppati
                if high_corr:
                    correlations.append("\nCorrelazioni forti:")
                    correlations.extend(high_corr)
                if medium_corr:
                    correlations.append("\nCorrelazioni moderate:")
                    correlations.extend(medium_corr)
                if low_corr:
                    correlations.append("\nCorrelazioni deboli:")
                    correlations.extend(low_corr)
            else:
                if query:
                    correlations.append(f"Non ho trovato correlazioni significative tra i documenti per la query '{query}'.")
                else:
                    correlations.append("Non ho trovato correlazioni significative tra i documenti disponibili.")
            
            return "\n".join(correlations)
            
        except Exception as e:
            logger.error(f"Errore nell'analisi delle correlazioni: {str(e)}")
            return "Si è verificato un errore durante l'analisi delle correlazioni tra i documenti."

    def _handle_direct_query(self, query: str) -> str:
        """Gestisce la query in modo diretto quando l'agente fallisce."""
        try:
            logger.info(f"Tentativo di gestione diretta della query: {query}")
            # Prova prima con la ricerca diretta
            results = self.vector_store.search(query)
            if results:
                logger.info(f"Trovati {len(results)} risultati con la ricerca diretta")
                formatted_results = []
                for doc in results:
                    source = doc.metadata.get('source', 'Sconosciuta')
                    content = doc.page_content.strip()
                    formatted_results.append(f"Il documento '{source}' contiene:\n{content}")
                return "\n\n---\n\n".join(formatted_results)
            
            # Se non trova risultati, prova con l'estrazione del testo
            logger.info("Nessun risultato trovato con la ricerca diretta, tentativo di estrazione testo")
            return self.extract_document_text(query)
            
        except Exception as e:
            logger.error(f"Errore nella gestione diretta della query: {str(e)}")
            return "Mi dispiace, non sono riuscito a trovare una risposta pertinente nei documenti."

    def _get_welcome_message(self) -> str:
        """Restituisce il messaggio di benvenuto dell'agente."""
        return HELP_MESSAGES["welcome"]

    def get_welcome_message(self) -> str:
        """Restituisce il messaggio di benvenuto."""
        return self._get_welcome_message() 