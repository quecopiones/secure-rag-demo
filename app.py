import streamlit as st
import tempfile
import os

# Librerias usadas
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from presidio_analyzer import AnalyzerEngine, RecognizerResult

# --- Configuración de la Página ---
st.set_page_config(page_title="SecureDoc Chat", page_icon="🛡️")
st.title("🛡️ SecureDoc Chat: RAG con Seguridad")

# Sidebar para API Key
api_key = st.sidebar.text_input("Tu OpenAI API Key", type="password")

# --- Funciones Principales ---
def procesar_pdf(uploaded_file):
    """
    Carga y fragmenta el PDF.
    Aquí es donde se aplica la ingeniería de datos.
    """
    # Creamos un archivo temporal para que PyPDFLoader pueda leerlo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Tamaño del fragmento
        chunk_overlap=200,    # Solapamiento entre fragmentos
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(docs)
    
    st.info(f"Splits generados del archivo: {len(splits)}")


    os.remove(tmp_path) # Limpieza
    return splits

def crear_vector_store(splits, api_key):
    """
    Crea la base de datos vectorial (Embeddings).
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Crear el vectorstore usando Chroma
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings
    )
    return vectorstore

def validacion_de_seguridad(respuesta):
    """
    (Seguridad): Guardrail simple.
    Verifica si la respuesta es segura o si alucina.
    """
    # Extraemos el texto de AIMessage
    if hasattr(respuesta, "content"):
        texto = respuesta.content
    else:
        texto = str(respuesta)

    texto = texto.lower()

    # Usamos presidio-analyze para detectar datos sensibles
    analyzer = AnalyzerEngine()
    # Analiza si la respuesta tiene PII
    results = analyzer.analyze(text=texto, entities=["PERSON", "DATE_TIME"], language='en')

    # Si no encuentra ninguna respuesta
    if "no tengo información" in texto:
        return "⚠️ Advertencia: la respuesta indica falta de información." 
    
    # Revisar los resultados del análisis y censuramos si hay datos personales
    for result in results:
        if isinstance(result, RecognizerResult):
            if result.entity_type in ["PERSON", "DATE_TIME"]:
                return "⚠️ Advertencia: No puedo contestarte eso porque es un dato de Identificacion Personal."

    return texto

def devuelve_respuesta_segura(rag_chain, pregunta, enviado):
    """
    Obtiene la respuesta de la cadena RAG y la valida.
    """
    if pregunta:
        with st.spinner("Generando respuesta segura..."):
            #Invocar la cadena RAG para obtener la respuesta 
            result = rag_chain.invoke(pregunta)
            #Pasar la respuesta por la función de seguridad
            respuesta_final = validacion_de_seguridad(result)
                    
            if enviado:
                st.markdown("### Pregunta:")
                st.write(f"Preguntaste: {pregunta}")
                st.write(f"Respuesta : {respuesta_final}")

        return respuesta_final

# --- Flujo de la Aplicación ---

if api_key:
    
    uploaded_file = st.file_uploader("Sube tu documento (PDF)", type="pdf")

    if uploaded_file:

        if "vectorstore" not in st.session_state:
                with st.spinner("Analizando documento... (Chunking & Embedding)"):
                    # Ingesta
                    splits = procesar_pdf(uploaded_file)
                    
                    # Almacenamiento Vectorial
                    vectorstore = crear_vector_store(splits, api_key)
                    
                    # Guardamos en el estado de la sesión el vectorstore
                    st.session_state.vectorstore = vectorstore

                    # Obtenemos el retractor del vectorstore
                    retriever = st.session_state.vectorstore.as_retriever()
                    
                    # Configuración del Chat (RAG)
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)
                    
                    # Crear el prompt
                    prompt = ChatPromptTemplate.from_messages([
                        ("system",
                        "Eres un asistente que contesta usando exclusivamente el contexto proporcionado."
                        ),
                        ("human",
                        "Pregunta: {pregunta_usuario}\n\n"
                        "Contexto recuperado:\n{context}"
                        )
                    ])


                    # 6. Crear la cadena de recuperación usando la nueva implementación de RAG de LangChain
                    rag_chain = (
                        {
                            "context": retriever,
                            "pregunta_usuario": RunnablePassthrough()
                        }
                        | prompt
                        | llm
                    )
                
                    # Guardamos en el estado de la sesión la cadena RAG
                    st.session_state.rag_chain = rag_chain

                st.success("¡Documento procesado y seguro! Pregúntame.")

        else:
            st.info("Documento ya procesado. Puedes hacer tus preguntas.")

        # Pregunta del usuario  
        with st.form("form_pregunta", clear_on_submit=True):
            pregunta = st.text_input("Escribe tu pregunta:")
            enviado = st.form_submit_button("Enviar")

        devuelve_respuesta_segura(st.session_state.rag_chain, pregunta, enviado)
    else:
        st.info("Por favor, sube un archivo PDF para comenzar.")
else:
    st.warning("Por favor, ingresa tu API Key para comenzar.")