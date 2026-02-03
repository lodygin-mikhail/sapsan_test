import hashlib
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.ingestion.docx_parser import DocxParser
from src.core.retriever import AsyncBM25Retriever
from src.core.vector_store import VectorStore


logger = logging.getLogger(__name__)

class IngestionService:
    """
    Сервис ingestion документов в RAG систему.
    """
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: Optional[AsyncBM25Retriever] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 300,
        add_start_index: bool = True,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
        )

    async def ingest_file(self, file_path: Path, filename: str) -> Dict:
        """
        Индексирует файл в векторное и BM25 хранилища.
        """
        logger.info("Начало ingestion файла: %s", filename)

        try:
            file_hash = self._calculate_file_hash(file_path)
            parser = DocxParser(str(file_path))
            docs = parser.parse()

            for d in docs:
                d.metadata.update(
                    {
                        "source": filename,
                        "file_hash": file_hash,
                        "total_pages": 1,
                    }
                )

            text_docs, table_docs = [], []
            for d in docs:
                if d.metadata.get("is_atomic"):
                    table_docs.append(d)
                else:
                    text_docs.append(d)

            chunks = self.splitter.split_documents(text_docs)
            chunks.extend(table_docs)

            documents, ids = self._prepare_chunks(
                chunks=chunks,
                filename=filename,
                file_hash=file_hash,
            )

            added_docs = await self.vector_store.aadd_documents(documents, ids)

            if self.bm25_retriever and added_docs:
                await self.bm25_retriever.aadd_documents(added_docs)

            logger.info(
                "Ingestion завершён: документов=%d, чанков=%d",
                len(docs),
                len(documents),
            )

            return {
                "documents": len(docs),
                "chunks": len(documents),
                "file_hash": file_hash,
                "parser": "DocxParser",
            }

        except Exception:
            logger.exception("Ошибка ingestion файла: %s", filename)
            raise

    # ---------- chunk preparation ----------
    def _prepare_chunks(
        self,
        chunks: List[Document],
        filename: str,
        file_hash: str,
    ):
        documents: List[Document] = []
        ids: List[str] = []

        for idx, chunk in enumerate(chunks):
            normalized_text = self._normalize_text(chunk.page_content)

            chunk_hash = self._chunk_hash(normalized_text)
            chunk_uuid = self._chunk_uuid(normalized_text)

            chunk.page_content = normalized_text

            chunk.metadata.update(
                {
                    "source": filename,
                    "file_hash": file_hash,
                    "chunk_hash": chunk_hash,
                    "chunk_index": idx,
                    "chunk_uuid": chunk_uuid,
                    "chunk_type": chunk.metadata.get("chunk_type", "paragraph"),
                    "section": chunk.metadata.get("section"),
                    "subsection": chunk.metadata.get("subsection"),
                }
            )

            documents.append(chunk)
            ids.append(chunk_uuid)

        return documents, ids

    # ---------- utils ----------
    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.replace("\xad", "").split())

    @staticmethod
    def _chunk_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _chunk_uuid(text: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

    @staticmethod
    def _calculate_file_hash(path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
