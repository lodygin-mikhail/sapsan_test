import asyncio
import logging
from typing import List

import torch
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            logger.warning("Используется CPU вместо GPU")
        self.load()

    def load(self):
        logger.debug("Загрузка модели %s", self.model_name)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.eval()
            logger.debug("Модель успешно загружена")
        except Exception as e:
            logger.error("Ошибка загрузки модели: %s", e, exc_info=True)
        self.model.to(self.device)
        logger.debug("Модель загружена в память")

    def unload(self):
        logger.debug("Выгрузка модели %s из памяти", self.model_name)
        try:
            self.model.to("cpu")
            self.model = None
            self.tokenizer = None
            logger.debug("Модель успешно выгружена")
        except Exception as e:
            logger.error("Ошибка при выгрузке модели: %s", e, exc_info=True)

    async def arerank(self, query: str, documents: List[Document]) -> List[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,  # ThreadPoolExecutor
            self._rerank_sync,
            query,
            documents,
        )

    def _rerank_sync(self, query: str, documents: List[Document]) -> List[Document]:
        if not self.model or not self.tokenizer:
            logger.error("Модель или токенизатор не загружены")
            return documents
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            # Для bge-reranker-v2-m3 score лежит в logits
            # shape: [batch_size, 1]
            scores = outputs.logits.squeeze(-1)
            scores = scores.detach().cpu().tolist()

            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            ranked_documents = [doc for doc, _ in scored_docs]

            return ranked_documents
