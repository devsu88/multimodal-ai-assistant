import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any, Optional, Union
import logging
import json
from langchain_core.language_models import BaseChatModel
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from vector_db.chroma_store import ChromaStore

from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            max_iterations=5,  # Ridotto a 5 per evitare loop
            early_stopping_method="force",
            return_intermediate_steps=True  # Aggiunto per debug
        )

    def _setup_tools(self) -> List[Tool]:
        """Definisce gli strumenti che l'agente può utilizzare."""
        return [
            Tool(
                name="list_documents",
                func=self.list_documents,
                description="Elenca i documenti presenti nel sistema. Usa questo strumento quando l'utente chiede quali documenti sono disponibili o quanti documenti ci sono."
            ),
            Tool(
                name="search_document_content",
                func=self.search_documents,
                description="Cerca informazioni specifiche nei documenti. Usa questo strumento quando l'utente fa domande sul contenuto dei documenti."
            ),
            Tool(
                name="extract_document_text",
                func=self.extract_document_text,
                description="Estrae il testo completo da un documento specifico. Usa questo strumento quando l'utente chiede di leggere il contenuto di un documento."
            )
        ]

    def _setup_prompt(self) -> ChatPromptTemplate:
        """Crea il prompt per l'agente."""
        template = """Sei un assistente AI specializzato nell'analisi di documenti. Il tuo compito è rispondere alle domande degli utenti basandoti esclusivamente sui documenti caricati.

REGOLE FONDAMENTALI:
1. Se non ci sono documenti caricati, informa cortesemente l'utente che è necessario caricarne almeno uno.
2. Rispondi SOLO basandoti sui documenti disponibili. Non inventare informazioni.
3. Se la domanda non è pertinente ai documenti, invita l'utente a porre una domanda attinente.
4. Quando possibile, indica la fonte delle informazioni (nome del documento).
5. Rispondi in italiano, con tono professionale ma accessibile.
6. Sii conciso e diretto nelle risposte. Evita di fare troppe riflessioni.
7. Rispondi SOLO a ciò che viene chiesto, non aggiungere informazioni non richieste.

Strumenti disponibili:
{tool_names}

Descrizione degli strumenti:
{tools}

ISTRUZIONI PER L'USO DEGLI STRUMENTI:
1. Usa 'list_documents' SOLO quando l'utente chiede:
   - quali documenti sono presenti
   - quanti documenti ci sono
   - elenca i documenti
   - mostra i file disponibili

2. Usa 'search_document_content' SOLO quando l'utente chiede:
   - informazioni specifiche che potrebbero essere in qualsiasi documento
   - cerca qualcosa in tutti i documenti
   - trova informazioni su un argomento specifico

3. Usa 'extract_document_text' SOLO quando l'utente chiede:
   - il contenuto di un file specifico
   - cosa contiene un documento particolare
   - leggi un file specifico
   - mostra il contenuto di un documento

IMPORTANTE: Scegli SEMPRE lo strumento più appropriato per la domanda specifica dell'utente.

Per usare uno strumento, usa questo formato:
```json
{{
    "action": "nome_strumento",
    "action_input": "input"
}}
```

Per la risposta finale, usa SEMPRE questo formato esatto:
```json
{{
    "action": "Final Answer",
    "action_input": "La tua risposta qui come stringa semplice"
}}
```

IMPORTANTE: La risposta finale DEVE essere una stringa semplice, NON un dizionario o un oggetto JSON.

Cronologia:
{chat_history}

Domanda: {input}

Riflessioni:
{agent_scratchpad}
"""
        return ChatPromptTemplate.from_template(template)

    def _is_relevant_query(self, query: str) -> bool:
        """Determina se la query è pertinente ai documenti."""
        # Parole chiave che indicano una domanda sui documenti
        document_keywords = [
            "documento", "documenti", "testo", "contenuto", "cerca", "trova",
            "cercare", "trovare", "leggi", "leggere", "analizza", "analizzare",
            "spiega", "spiegare", "descrivi", "descrivere", "riassumi", "riassumere",
            "cosa dice", "cosa dicono", "cosa contiene", "cosa contengono",
            "parla di", "parlano di", "tratta di", "trattano di",
            "informazioni", "dettagli", "particolari", "specifiche",
            "estrai", "leggi", "mostra", "contenuto", "testo"
        ]
        
        # Parole chiave che indicano una domanda generale
        general_keywords = [
            "ciao", "salve", "buongiorno", "buonasera", "come stai",
            "grazie", "prego", "aiuto", "aiutami", "cosa puoi fare",
            "funziona", "funzionare", "capire", "capisco", "non capisco"
        ]
        
        query_lower = query.lower()
        
        # Se la query contiene parole chiave relative ai documenti, è pertinente
        if any(keyword in query_lower for keyword in document_keywords):
            return True
            
        # Se la query contiene solo parole chiave generali, non è pertinente
        if any(keyword in query_lower for keyword in general_keywords):
            return False
            
        # Se non ci sono parole chiave specifiche, assumiamo che sia pertinente
        return True

    def _analyze_query(self, query: str) -> dict:
        """Analizza la query usando l'LLM per determinare il tipo e la pertinenza."""
        # Prepara il contesto della conversazione
        chat_history = self.memory.chat_memory.messages if self.memory else []
        context = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history]) if chat_history else ""

        prompt = f"""Analizza la seguente domanda dell'utente e determina:
1. Se è una domanda generale (non relativa ai documenti)
2. Se è pertinente ai documenti
3. Se è una richiesta di sistema (es. aiuto, informazioni sul sistema)
4. Se è una domanda di follow-up (continua una conversazione precedente)
5. Se è una richiesta di riassunto (es. "riassumi", "sintetizza", "in poche parole")

REGOLE PER L'ANALISI:
- Una domanda è pertinente se:
  * Chiede di cercare qualcosa nei documenti
  * Chiede se un argomento è presente nei documenti
  * Chiede informazioni sul contenuto dei documenti
  * Chiede di elencare i documenti disponibili
  * È una domanda di follow-up che continua una ricerca precedente
  * Chiede un riassunto o una sintesi di un documento
- Una domanda è generale se:
  * Non fa riferimento ai documenti
  * Chiede informazioni generali non relative ai documenti
  * È un saluto o una conversazione casuale
- Una domanda è di sistema se:
  * Chiede aiuto o istruzioni
  * Chiede informazioni sul sistema
  * Chiede come funziona l'assistente
- Una domanda è di riassunto se:
  * Contiene parole come "riassumi", "sintetizza", "in poche parole"
  * Chiede un riassunto di un documento
  * Specifica un limite di parole o caratteri

Esempi di domande pertinenti:
- "Si parla di calcio nei documenti?"
- "Cerca informazioni sul calcio"
- "Quali documenti sono presenti?"
- "Cosa contiene il file X?"
- "E di competenze tecniche?" (come follow-up)
- "Riassumi il documento in 100 parole"

Esempi di domande generali:
- "Come stai?"
- "Cosa ne pensi del calcio?"
- "Qual è il tuo colore preferito?"

Esempi di domande di sistema:
- "Come funziona?"
- "Aiuto"
- "Info"

Esempi di domande di riassunto:
- "Riassumi il documento"
- "Sintetizza in 100 parole"
- "Fammi un riassunto breve"

Cronologia completa della conversazione:
{context}

Domanda da analizzare: {query}

Rispondi in formato JSON con i seguenti campi:
{{
    "is_general": true/false,
    "is_relevant": true/false,
    "is_system": true/false,
    "is_follow_up": true/false,
    "is_summary": true/false,
    "word_limit": number/null,
    "reason": "breve spiegazione della decisione"
}}"""

        try:
            response = self.llm.invoke(prompt)
            # Estrai il JSON dalla risposta
            json_str = response.content
            if isinstance(json_str, str):
                # Rimuovi eventuali markdown o altri caratteri
                json_str = json_str.replace("```json", "").replace("```", "").strip()
                return json.loads(json_str)
            return json_str
        except Exception as e:
            logger.error(f"Errore nell'analisi della query: {str(e)}")
            # In caso di errore, assumiamo che la query sia pertinente
            return {
                "is_general": False,
                "is_relevant": True,
                "is_system": False,
                "is_follow_up": False,
                "is_summary": False,
                "word_limit": None,
                "reason": "Errore nell'analisi, default a pertinente"
            }

    def process_query(self, query: str) -> str:
        """Elabora una query dell'utente e restituisce una risposta."""
        try:
            # Verifica se ci sono documenti
            if not self.has_documents():
                analysis = self._analyze_query(query)
                if analysis["is_general"]:
                    return "Ciao! Sono il tuo assistente per l'analisi dei documenti. Per poterti aiutare, ho bisogno che tu carichi almeno un documento."
                return "Per poterti aiutare, ho bisogno che tu carichi almeno un documento."

            # Analizza la query
            analysis = self._analyze_query(query)
            
            # Gestione query di sistema
            if analysis["is_system"]:
                return self._handle_system_query(query)

            # Verifica se la query è pertinente
            if not analysis["is_relevant"] and not analysis["is_follow_up"]:
                return "Per rispondere, ho bisogno che la tua domanda sia riferita ai documenti che hai caricato. Vuoi chiedermi qualcosa in merito a essi?"

            # Verifica se è una richiesta di riassunto
            if analysis["is_summary"]:
                # Estrai il contenuto del documento
                content = self.extract_document_text(query)
                return self._handle_summary_request(query, content)

            # Verifica se è una richiesta di estrazione del testo
            if any(keyword in query.lower() for keyword in ["estrai", "leggi", "mostra", "contenuto", "testo"]):
                return self.extract_document_text(query)

            # Elaborazione con l'agente
            try:
                # Passa l'intera cronologia della conversazione
                response = self.agent_executor.invoke({
                    "input": query,
                    "chat_history": self.memory.chat_memory.messages if self.memory else []
                })
                
                # Verifica se ci sono stati errori o interruzioni
                if "Agent stopped" in str(response.get("output", "")):
                    # Prova un approccio più diretto
                    return self._handle_direct_query(query)
                
                output = response.get("output", "")
                # Se l'output è un dizionario, estrai la risposta
                if isinstance(output, dict) and "risposta" in output:
                    output = output["risposta"]
                
                return self._clean_response(output)
                
            except Exception as e:
                logger.error(f"Errore nell'esecuzione dell'agente: {str(e)}")
                return self._handle_direct_query(query)
            
        except Exception as e:
            logger.error(f"Errore nell'elaborazione della query: {str(e)}", exc_info=True)
            return "Mi dispiace, si è verificato un errore. Puoi riprovare con una domanda diversa?"

    def _is_general_query(self, query: str) -> bool:
        """Determina se la query è generale e non richiede documenti."""
        general_keywords = [
            "ciao", "salve", "buongiorno", "buonasera", "come stai",
            "grazie", "prego", "aiuto", "aiutami", "cosa puoi fare",
            "funziona", "funzionare", "capire", "capisco", "non capisco"
        ]
        return any(keyword in query.lower() for keyword in general_keywords)

    def _clean_response(self, response: str) -> str:
        """Pulisce la risposta da formattazioni non necessarie."""
        if not response:
            return "Mi dispiace, non sono riuscito a generare una risposta."
        
        # Rimuovi formattazioni non necessarie
        response = response.strip()
        response = response.replace('"', '')
        response = response.replace('```', '')
        
        return response.strip()

    def has_documents(self) -> bool:
        """Verifica se ci sono documenti nel database."""
        try:
            return self.vector_store.count() > 0
        except Exception as e:
            logger.error(f"Errore nel conteggio dei documenti: {str(e)}")
            return False

    def _get_general_help(self) -> str:
        """Fornisce informazioni generali sull'assistente."""
        return """Ciao! Sono il tuo assistente per l'analisi dei documenti. Posso aiutarti a:
- Cercare informazioni nei documenti
- Estrarre il testo dai documenti
- Elencare i documenti disponibili
- Rispondere a domande sul contenuto dei documenti

Per iniziare, carica almeno un documento e fammi una domanda!"""

    def _get_system_info(self) -> str:
        """Fornisce informazioni sul sistema."""
        return """Sistema di Analisi Documenti
- Supporta documenti PDF e immagini
- Utilizza OCR per estrarre testo dalle immagini
- Permette ricerche semantiche nel contenuto
- Mantiene la cronologia delle conversazioni

Per iniziare, carica un documento e fammi una domanda!"""

    def _get_usage_help(self) -> str:
        """Fornisce istruzioni d'uso dettagliate."""
        return """Ecco come utilizzare l'assistente:

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

    def _handle_system_query(self, query: str) -> Optional[str]:
        """Gestisce le query di sistema."""
        query = query.lower().strip()
        
        # Usa l'LLM per determinare se è una richiesta di aiuto
        prompt = f"""Determina se questa è una richiesta di aiuto o informazioni sul sistema:
{query}

Rispondi in formato JSON:
{{
    "is_help": true/false,
    "help_type": "general/system/usage" o null
}}"""

        try:
            response = self.llm.invoke(prompt)
            analysis = json.loads(response.content)
            
            if analysis.get("is_help"):
                help_type = analysis.get("help_type", "general")
                if help_type == "general":
                    return self._get_general_help()
                elif help_type == "system":
                    return self._get_system_info()
                elif help_type == "usage":
                    return self._get_usage_help()
            
            return None
        except Exception as e:
            logger.error(f"Errore nell'analisi della query di sistema: {str(e)}")
            return None

    def list_documents(self, _: Optional[str] = None) -> str:
        """Elenca i documenti presenti nel database."""
        try:
            if not self.has_documents():
                return "Non ci sono documenti caricati nel sistema."

            sources = self.vector_store.get_document_sources()
            num_docs = len(sources)

            if num_docs == 1:
                return f"Attualmente c'è 1 documento nel sistema: {sources[0]}"
            
            doc_list = "\n - ".join(sources)
            return f"Attualmente ci sono {num_docs} documenti nel sistema:\n - {doc_list}"
            
        except Exception as e:
            logger.error(f"Errore nell'elencazione dei documenti: {str(e)}")
            return "Si è verificato un errore nel recupero della lista dei documenti."

    def search_documents(self, query: str) -> str:
        """Cerca informazioni nei documenti."""
        try:
            if not self.has_documents():
                return "Non ci sono documenti disponibili per la ricerca."

            results = self.vector_store.search(query)
            
            if not results:
                return "Non ho trovato informazioni pertinenti nei documenti per la tua domanda."

            formatted_results = []
            for doc in results:
                source = doc.metadata.get('source', 'Sconosciuta')
                content = doc.page_content.strip()
                formatted_results.append(f"Secondo il documento '{source}':\n{content}")
            
            return "\n\n---\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Errore nella ricerca dei documenti: {str(e)}")
            return "Si è verificato un errore durante la ricerca nei documenti."

    def extract_document_text(self, query: str) -> str:
        """Estrae il testo da un documento specifico o dall'ultimo caricato."""
        try:
            if not self.has_documents():
                return "Non ci sono documenti disponibili per l'estrazione del testo."

            # Determina quale documento estrarre
            if "ultimo" in query.lower() or "last" in query.lower():
                # Estrai dall'ultimo documento
                sources = self.vector_store.get_document_sources()
                if not sources:
                    return "Non ci sono documenti disponibili."
                target_source = sources[-1]
            else:
                # Estrai il nome del file dalla query
                import re
                patterns = [
                    r"file\s+(\S+)",  # "file nome.pdf"
                    r"documento\s+(\S+)",  # "documento nome.pdf"
                    r"(\S+\.(pdf|png|jpg|jpeg))",  # "nome.pdf" o "nome.png"
                    r"(\S+)$"  # ultima parola della query
                ]
                
                target_source = None
                for pattern in patterns:
                    match = re.search(pattern, query.lower())
                    if match:
                        target_source = match.group(1)
                        break
                
                if not target_source:
                    return "Non ho capito quale documento vuoi che analizzi. Puoi specificare il nome del file?"

            # Cerca il documento nel database
            sources = self.vector_store.get_document_sources()
            matching_sources = [s for s in sources if target_source.lower() in s.lower()]
            
            if not matching_sources:
                return f"Nessun documento trovato che contenga '{target_source}' nel nome. Documenti disponibili:\n" + "\n".join(f"- {s}" for s in sources)
            
            if len(matching_sources) > 1:
                return f"Ho trovato più documenti che corrispondono a '{target_source}':\n" + "\n".join(f"- {s}" for s in matching_sources) + "\n\nPuoi specificare meglio quale documento vuoi analizzare?"
            
            # Estrai il testo dal documento trovato
            results = self.vector_store.search_by_source(matching_sources[0])
            
            if not results:
                return f"Non sono riuscito a estrarre il testo dal documento '{matching_sources[0]}'."

            # Estrai e formatta il testo
            text_parts = []
            for doc in results:
                text_parts.append(doc.page_content.strip())
            
            full_text = "\n\n".join(text_parts)
            return f"Contenuto del documento '{matching_sources[0]}':\n\n{full_text}"
            
        except Exception as e:
            logger.error(f"Errore nell'estrazione del testo: {str(e)}")
            return "Si è verificato un errore durante l'estrazione del testo dal documento."

    def _handle_direct_query(self, query: str) -> str:
        """Gestisce la query in modo diretto quando l'agente fallisce."""
        try:
            # Prova prima con la ricerca diretta
            results = self.vector_store.search(query)
            if results:
                formatted_results = []
                for doc in results:
                    source = doc.metadata.get('source', 'Sconosciuta')
                    content = doc.page_content.strip()
                    formatted_results.append(f"Secondo il documento '{source}':\n{content}")
                return "\n\n---\n\n".join(formatted_results)
            
            # Se non trova risultati, prova con l'estrazione del testo
            return self.extract_document_text(query)
            
        except Exception as e:
            logger.error(f"Errore nella gestione diretta della query: {str(e)}")
            return "Mi dispiace, non sono riuscito a trovare una risposta pertinente nei documenti."

    def _get_welcome_message(self) -> str:
        """Restituisce il messaggio di benvenuto dell'agente."""
        return """👋 Benvenuto! Sono il tuo assistente AI specializzato nell'analisi dei documenti.

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

Sono qui per rendere più semplice e veloce l'analisi dei tuoi documenti. Come posso aiutarti oggi? 😊"""

    def _initialize_agent(self):
        """Inizializza l'agente con gli strumenti necessari."""
        # Inizializza la memoria
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Definisci gli strumenti disponibili
        tools = [
            Tool(
                name="search_documents",
                func=self._search_documents,
                description="Cerca informazioni nei documenti caricati"
            ),
            Tool(
                name="extract_text",
                func=self.extract_document_text,
                description="Estrae e legge il contenuto di un documento"
            ),
            Tool(
                name="list_documents",
                func=self._list_documents,
                description="Elenca i documenti disponibili"
            )
        ]

        # Crea l'agente
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self._create_agent(tools),
            tools=tools,
            memory=self.memory,
            verbose=True
        )

    def get_welcome_message(self) -> str:
        """Restituisce il messaggio di benvenuto."""
        return self._get_welcome_message()

    def _handle_summary_request(self, query: str, content: str) -> str:
        """Gestisce una richiesta di riassunto."""
        try:
            prompt = f"""Riassumi il seguente testo in modo conciso e chiaro.

Testo da riassumere:
{content}

Regole per il riassunto:
1. Cattura SOLO i punti principali
2. Evita ripetizioni
3. Usa frasi brevi e dirette
4. Organizza le informazioni in modo logico
5. Evita dettagli non essenziali
6. Non includere esempi o casi specifici
7. Non ripetere la stessa informazione in modi diversi

Formato del riassunto:
- Inizia con una frase che introduce l'argomento principale
- Continua con i punti chiave in ordine di importanza
- Concludi con una frase riassuntiva"""

            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"Errore nella generazione del riassunto: {str(e)}")
            return "Mi dispiace, non sono riuscito a generare un riassunto del documento." 