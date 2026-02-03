import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


logger = logging.getLogger(__name__)

class BaseRetriever(ABC):
    @abstractmethod
    async def aretrieve(self, query: str) -> List[Document]:
        pass


class VectorRetriever(BaseRetriever):
    """
    Ретривер, выполняющий поиск документов в векторном хранилище.
    """

    def __init__(self, vector_store, k: int = 5, fetch_k: int = 50):
        self.vector_store = vector_store
        self.k = k
        self.fetch_k = fetch_k

        logger.info(
            "VectorRetriever инициализирован (k=%d, fetch_k=%d)",
            k,
            fetch_k,
        )

    async def aretrieve(self, query: str) -> List[Document]:
        """
        Возвращает релевантные документы из векторного хранилища.
        """
        logger.debug("VectorRetriever: поиск документов по запросу")

        try:
            return await self.vector_store.aread(
                query=query,
                k=self.k,
                fetch_k=self.fetch_k,
            )
        except Exception:
            logger.exception("Ошибка поиска в VectorStore")
            raise



class AsyncBM25Retriever(BaseRetriever):
    """
    Асинхронная обёртка над BM25Retriever.
    """

    def __init__(self, documents: list[Document] | None = None):
        self._lock = asyncio.Lock()
        self._documents: list[Document] = documents or []
        self.retriever: BM25Retriever | None = (
            BM25Retriever.from_documents(self._documents)
            if self._documents
            else None
        )

        logger.info(
            "AsyncBM25Retriever инициализирован, документов: %d",
            len(self._documents),
        )

    async def aretrieve(self, query: str) -> list[Document]:
        if not self.retriever:
            logger.warning("BM25Retriever не инициализирован")
            return []

        async with self._lock:
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.retriever.invoke(query)
                )
            except Exception:
                logger.exception("Ошибка BM25 retrieval")
                raise

    async def aadd_documents(self, new_docs: list[Document]) -> None:
        if not new_docs:
            logger.debug("BM25: нет новых документов")
            return

        async with self._lock:
            try:
                loop = asyncio.get_running_loop()

                def _update():
                    self._documents.extend(new_docs)
                    self.retriever = BM25Retriever.from_documents(
                        self._documents
                    )

                await loop.run_in_executor(None, _update)

                logger.info(
                    "BM25 индекс обновлён, всего документов: %d",
                    len(self._documents),
                )

            except Exception:
                logger.exception("Ошибка обновления BM25 индекса")
                raise



class HybridRetriever(BaseRetriever):
    """
    Гибридный ретривер: Vector + BM25 + rerank.
    """

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        reranker,
        pre_rerank_k: int = 30,
        top_k: int = 5,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.pre_rerank_k = pre_rerank_k
        self.top_k = top_k

        logger.info(
            "HybridRetriever инициализирован (pre_rerank_k=%d, top_k=%d)",
            pre_rerank_k,
            top_k,
        )

    async def aretrieve(self, query: str) -> list[Document]:
        """
        Выполняет гибридный поиск и rerank документов.
        """
        logger.info("HybridRetriever: запуск retrieval")

        try:
            vec_docs = await self.vector_retriever.aretrieve(query)
            logger.debug("VectorRetriever вернул документов: %d", len(vec_docs))

            bm25_docs = (
                await self.bm25_retriever.aretrieve(query)
                if self.bm25_retriever
                else []
            )

            logger.debug("BM25Retriever вернул документов: %d", len(bm25_docs))

            docs_map = {}
            for doc in vec_docs + bm25_docs:
                key = doc.metadata.get("chunk_hash") or doc.page_content
                docs_map[key] = doc

            merged_docs = list(docs_map.values())
            candidates = merged_docs[:self.pre_rerank_k]

            logger.debug(
                "Кандидатов перед rerank: %d",
                len(candidates),
            )

            ranked_docs = await self.reranker.arerank(query, candidates)

            logger.info(
                "HybridRetriever завершён, возвращено документов: %d",
                min(len(ranked_docs), self.top_k),
            )

            return ranked_docs[:self.top_k]

        except Exception:
            logger.exception("Ошибка HybridRetriever")
            raise
