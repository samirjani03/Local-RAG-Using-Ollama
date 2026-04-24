from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_UPLOAD_TYPES = ("pdf", "txt", "csv", "docx")


@dataclass(slots=True)
class FileIngestionResult:
    filename: str
    file_type: str
    document_count: int = 0
    chunk_count: int = 0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class IngestionBatchResult:
    chunks: list[Document]
    files: list[FileIngestionResult]


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )


def ingest_file_paths(file_paths: Iterable[str | Path]) -> IngestionBatchResult:
    text_splitter = create_text_splitter()
    all_chunks: list[Document] = []
    results: list[FileIngestionResult] = []

    for raw_path in file_paths:
        path = Path(raw_path)
        file_type = path.suffix.lower().lstrip(".")
        result = FileIngestionResult(filename=path.name, file_type=file_type or "unknown")

        try:
            documents = load_documents_from_path(path)
            result.document_count = len(documents)

            if not documents:
                result.error = "No readable content was extracted from this file."
            else:
                chunks = text_splitter.split_documents(documents)
                result.chunk_count = len(chunks)
                if not chunks:
                    result.error = "The extracted content could not be split into chunks."
                else:
                    all_chunks.extend(chunks)
        except Exception as exc:
            result.error = str(exc)

        results.append(result)

    return IngestionBatchResult(chunks=all_chunks, files=results)


def load_documents_from_path(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    loader = get_loader_for_path(path)
    documents = loader.load()
    return [attach_source_metadata(document, path) for document in documents]


def get_loader_for_path(file_path: Path):
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(str(file_path))
    if suffix == ".txt":
        return TextLoader(str(file_path), autodetect_encoding=True)
    if suffix == ".csv":
        return CSVLoader(str(file_path), autodetect_encoding=True)
    if suffix == ".docx":
        ensure_docx_support()
        return Docx2txtLoader(str(file_path))

    raise ValueError(
        f"Unsupported file type '{suffix or '[no extension]'}'. "
        f"Supported types: {', '.join(f'.{file_type}' for file_type in SUPPORTED_UPLOAD_TYPES)}."
    )


def ensure_docx_support() -> None:
    if importlib.util.find_spec("docx2txt") is None:
        raise ImportError(
            "DOCX support requires the 'docx2txt' package. Install it with "
            "'.\\.venv\\Scripts\\python.exe -m pip install docx2txt'."
        )


def attach_source_metadata(document: Document, file_path: Path) -> Document:
    metadata = dict(document.metadata)
    metadata["source"] = file_path.name
    metadata["file_name"] = file_path.name
    metadata["file_type"] = file_path.suffix.lower().lstrip(".")
    return Document(page_content=document.page_content, metadata=metadata)
