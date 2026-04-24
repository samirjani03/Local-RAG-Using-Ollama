import os
import tempfile
from pathlib import Path

import streamlit as st

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings

from src.engine import EMBEDDING_MODEL, LLM_MODEL, get_conversational_rag_chain
from src.ingest import SUPPORTED_UPLOAD_TYPES, ingest_file_paths

PERSIST_DIRECTORY = "./chroma_db"

st.set_page_config(page_title="Local RAG App", page_icon="📚", layout="wide")

st.title("📚 Local PDF RAG Assistant")
st.markdown(f"**Powered by Ollama ({LLM_MODEL}) and ChromaDB**")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

embeddings = load_embeddings()

def init_vectorstore():
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        try:
            return Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
            )
        except Exception as exc:
            st.warning(f"Could not load existing vector DB: {exc}")
            return None
    return None


def vectorstore_has_documents(vectorstore: Chroma | None) -> bool:
    if vectorstore is None:
        return False

    try:
        stored_items = vectorstore.get(limit=1, include=[])
        return bool(stored_items.get("ids"))
    except Exception:
        return False


def process_uploaded_documents(uploaded_files) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_paths: list[Path] = []
        for index, uploaded_file in enumerate(uploaded_files):
            original_name = Path(uploaded_file.name).name
            file_path = Path(temp_dir) / original_name
            if file_path.exists():
                file_path = (
                    Path(temp_dir)
                    / f"{Path(original_name).stem}_{index}{Path(original_name).suffix}"
                )
            file_path.write_bytes(uploaded_file.getbuffer())
            temp_paths.append(file_path)

        batch_result = ingest_file_paths(temp_paths)

    successful_files = [item for item in batch_result.files if item.succeeded]
    failed_files = [item for item in batch_result.files if not item.succeeded]

    if batch_result.chunks:
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = Chroma.from_documents(
                documents=batch_result.chunks,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY,
            )
        else:
            st.session_state.vectorstore.add_documents(batch_result.chunks)

        st.success(
            f"Processed {len(successful_files)} file(s) into {len(batch_result.chunks)} chunks. "
            "Vector DB updated."
        )
    else:
        st.error("No usable document content was ingested from the selected files.")

    if successful_files:
        st.info(
            "\n".join(
                f"- {item.filename}: {item.document_count} document(s), {item.chunk_count} chunk(s)"
                for item in successful_files
            )
        )

    if failed_files:
        st.warning(
            "\n".join(f"- {item.filename}: {item.error}" for item in failed_files)
        )


if st.session_state.vectorstore is None:
    st.session_state.vectorstore = init_vectorstore()

with st.sidebar:
    st.header("Document Upload")
    st.markdown("Upload PDFs, TXT files, CSVs, or DOCX files in one batch.")
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=list(SUPPORTED_UPLOAD_TYPES),
        accept_multiple_files=True,
    )

    if st.button("Process Documents", disabled=not uploaded_files) and uploaded_files:
        with st.spinner("Processing documents... This may take a while for large files."):
            try:
                process_uploaded_documents(uploaded_files)
            except Exception as exc:
                st.error(f"Error processing documents: {exc}")

    if st.button("Clear Vector Database"):
        try:
            active_vectorstore = st.session_state.vectorstore

            if active_vectorstore is None and os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
                active_vectorstore = Chroma(
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=embeddings,
                )

            if active_vectorstore is None:
                st.info("No vector database found.")
            else:
                active_vectorstore.reset_collection()
                st.session_state.vectorstore = active_vectorstore
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success("Vector database cleared.")
        except Exception as exc:
            st.error(f"Could not clear the vector database: {exc}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not vectorstore_has_documents(st.session_state.vectorstore):
        with st.chat_message("assistant"):
            msg = (
                "Please upload and process documents first, or ensure your persistent "
                "DB is loaded."
            )
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    rag_chain = get_conversational_rag_chain(st.session_state.vectorstore)
                    response = rag_chain.invoke(
                        {
                            "input": prompt,
                            "chat_history": st.session_state.chat_history,
                        }
                    )

                    answer = response["answer"]
                    st.markdown(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state.chat_history.extend(
                        [
                            HumanMessage(content=prompt),
                            AIMessage(content=answer),
                        ]
                    )
                except Exception as exc:
                    st.error(f"Error generating response: {exc}")
