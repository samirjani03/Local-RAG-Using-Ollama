import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Configuration
PERSIST_DIRECTORY = "./chroma_db"
LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text" # A standard highly efficient local embedding model

st.set_page_config(page_title="Local RAG App", page_icon="📚", layout="wide")

st.title("📚 Local PDF RAG Assistant")
st.markdown(f"**Powered by Ollama ({LLM_MODEL}) and ChromaDB**")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

embeddings = load_embeddings()

def init_vectorstore():
    # Load existing vectorstore if it exists
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        try:
            return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        except Exception as e:
            st.warning(f"Could not load existing vector DB: {e}")
            return None
    return None

if st.session_state.vectorstore is None:
    st.session_state.vectorstore = init_vectorstore()

# Sidebar for PDF upload and processing
with st.sidebar:
    st.header("Document Upload")
    st.markdown("Upload your 200-300 page PDFs here.")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Process Document") and uploaded_file is not None:
        with st.spinner("Processing PDF... This may take a while for large documents."):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                
            try:
                # 1. Extract Text
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                # 2. Chunk Text (Ideal sizes for long context)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_documents(documents)
                
                if not chunks:
                    st.error("No text could be extracted from this PDF. It might be a scanned document or an image-based PDF.")
                else:
                    st.info(f"Split document into {len(chunks)} chunks.")
                    
                    # 3. Create Vector Store and Persist
                    # If persistent DB already exists, from_documents will add to it
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=PERSIST_DIRECTORY
                    )
                    st.success("Document processed and embedded successfully! Vector DB updated.")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    if st.button("Clear Vector Database"):
        import shutil
        if os.path.exists(PERSIST_DIRECTORY):
            # To avoid file lock issues in Windows, we try to clear the Chroma collection 
            # or just delete the folder manually. Deleting folder is easier:
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                st.session_state.vectorstore = None
                st.success("Vector database cleared. Please reload the app.")
            except Exception as e:
                st.error(f"Could not completely delete the folder due to file locks: {e}")
        else:
            st.info("No vector database found.")

# Main Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            msg = "Please upload and process a document first, or ensure your persistent DB is loaded."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Initialize LLM
                    llm = OllamaLLM(model=LLM_MODEL)
                    
                    # Create prompt template
                    system_prompt = (
                        "You are a helpful AI assistant. Use the following retrieved context "
                        "to answer the question. If you don't know the answer, say that you don't know. "
                        "Keep your answer concise and relevant.\n\n"
                        "Context: {context}"
                    )
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    # Create chains
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    
                    # Get response
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
