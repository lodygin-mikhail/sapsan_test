import logging
from contextlib import asynccontextmanager

from langchain_community.retrievers import BM25Retriever

from src.core.config import settings
from src.core.embedder import Embedder
from src.core.ingestion.ingestion import IngestionService
from src.core.llm_factory import get_llm
from src.core.logger import setup_logger
from src.core.pipeline import RAGPipeline
from src.core.reranker import Reranker
from src.core.retriever import VectorRetriever, AsyncBM25Retriever, HybridRetriever
from src.core.vector_store import VectorStore


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    """
    Lifespan-контекст приложения.

    Отвечает за инициализацию всех основных компонентов RAG:
    эмбеддер, векторное хранилище, ретриверы, пайплайн.
    """
    logger.info("Запуск lifespan приложения")

    try:
        setup_logger()
        logger.info("Логгер успешно инициализирован")

        embedder = Embedder()
        logger.info("Embedder инициализирован")

        vector_store = VectorStore(embeddings=embedder.client)
        await vector_store.ainit_collection()
        vector_store.init_vector_store()
        logger.info("VectorStore готов к работе")

        all_docs = await vector_store.aget_all_documents()
        logger.info("Загружено документов из векторного хранилища: %d", len(all_docs))

        llm = get_llm()
        reranker = Reranker(settings.RERANKER_MODEL)

        vector_retriever = VectorRetriever(vector_store=vector_store)

        bm25 = (
            AsyncBM25Retriever(
                retriever=BM25Retriever.from_documents(all_docs)
            )
            if all_docs
            else None
        )

        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25,
            reranker=reranker,
        )

        pipeline = RAGPipeline(
            llm=llm,
            retriever=vector_retriever,
        )

        ingest_service = IngestionService(
            vector_store=vector_store,
            bm25_retriever=bm25,
        )

        app.state.pipeline = pipeline
        app.state.vector_store = vector_store
        app.state.reranker = reranker
        app.state.retriever = hybrid_retriever
        app.state.ingest_service = ingest_service

        logger.info("Все сервисы успешно инициализированы")

        yield

    except Exception:
        logger.exception("Ошибка во время lifespan инициализации")
        raise

    finally:
        try:
            reranker.unload()
            logger.info("Reranker выгружен")
        except Exception:
            logger.warning("Ошибка при выгрузке reranker", exc_info=True)
