import logging

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, models

from src.core.config import settings


logger = logging.getLogger(__name__)

class VectorStore:
    """
    Обёртка над Qdrant для асинхронной работы.
    """
    def __init__(self, embeddings):
        """
        Инициализирует клиент Qdrant и сохраняет объект эмбеддингов.
        """
        logger.info("Инициализация VectorStore")

        try:
            self.client = AsyncQdrantClient(
                url=settings.QDRANT_URL
            )
            self.embeddings = embeddings
            self.store = None

            logger.info(
                "Подключение к Qdrant инициализировано: %s",
                settings.QDRANT_URL,
            )

        except Exception:
            logger.exception("Ошибка инициализации клиента Qdrant")
            raise

    async def ainit_collection(self, vector_size: int = 3072) -> None:
        """
        Проверяет существование коллекции и создаёт её при отсутствии.
        """
        logger.info(
            "Проверка существования коллекции Qdrant: %s",
            settings.COLLECTION_NAME,
        )

        try:
            exists = await self.client.collection_exists(
                settings.COLLECTION_NAME
            )

            if exists:
                logger.info(
                    "Коллекция Qdrant уже существует: %s",
                    settings.COLLECTION_NAME,
                )
                return

            logger.info(
                "Коллекция не найдена, создаём новую: %s",
                settings.COLLECTION_NAME,
            )

            await self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

            logger.info(
                "Коллекция Qdrant успешно создана: %s",
                settings.COLLECTION_NAME,
            )

        except Exception:
            logger.exception("Ошибка инициализации коллекции Qdrant")
            raise

    def init_vector_store(self):
        """
        Инициализирует LangChain-обёртку над существующей коллекцией.
        """
        try:
            self.store = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=settings.COLLECTION_NAME,
                url=settings.QDRANT_URL,
            )
            logger.info("VectorStore успешно инициализирован")
        except Exception:
            logger.exception("Ошибка инициализации VectorStore")
            raise

    async def aread(self, query: str, k: int = 5, fetch_k: int = 50):
        """
        Выполняет MMR-поиск документов в векторном хранилище.
        """
        logger.debug(
            "VectorStore search (k=%d, fetch_k=%d)",
            k,
            fetch_k,
        )

        try:
            retriever = self.store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k,
                }
            )
            return await retriever.ainvoke(query)

        except Exception:
            logger.exception("Ошибка чтения из VectorStore")
            raise

    async def aadd_documents(
        self,
        documents: list[Document],
        ids: list[str],
    ):
        """
        Добавляет документы в векторное хранилище.
        """
        logger.info("Добавление документов в VectorStore: %d", len(documents))

        try:
            await self.store.aadd_documents(
                documents=documents,
                ids=ids,
            )

            return documents
        except Exception:
            logger.exception("Ошибка добавления документов в VectorStore")
            raise

    async def aget_all_documents(self, batch_size: int = 1000) -> list[Document]:
        """
        Выгружает все документы из коллекции Qdrant.
        """
        logger.info("Загрузка всех документов из VectorStore")

        offset = None
        all_docs: list[Document] = []

        try:
            while True:
                points, offset = await self.client.scroll(
                    collection_name=settings.COLLECTION_NAME,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                )

                if not points:
                    break

                for point in points:
                    payload = point.payload or {}

                    doc = Document(
                        page_content=(
                                payload.get("page_content")
                                or payload.get("text")
                                or ""
                        ),
                        metadata=payload,
                    )
                    all_docs.append(doc)

                if offset is None:
                    break

            logger.info(
                "Загрузка документов завершена, всего: %d",
                len(all_docs),
            )

            return all_docs

        except Exception:
            logger.exception("Ошибка получения документов из VectorStore")
            raise