import logging
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from src.core.prompt import RAG_PROMPT


logger = logging.getLogger(__name__)

@dataclass
class RAGOutput:
    question: str
    answer: str
    documents: List[Document]
    contexts: List[str]


class RAGPipeline:
    """
    Основной RAG пайплайн: retrieval + generation.
    """

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    async def arun(self, query: str):
        logger.info("Запуск RAG pipeline")

        try:
            docs = await self.retriever.aretrieve(query)
            contexts = [d.page_content for d in docs]

            prompt = RAG_PROMPT.format_messages(
                question=query,
                context="\n\n".join(contexts),
            )

            response = await self.llm.ainvoke(prompt)

            logger.info("RAG pipeline успешно завершён")

            return RAGOutput(
                question=query,
                answer=response.content,
                documents=docs,
                contexts=contexts,
            )

        except Exception:
            logger.exception("Ошибка выполнения RAG pipeline")
            raise