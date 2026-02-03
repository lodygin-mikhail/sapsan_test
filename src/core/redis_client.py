from redis.asyncio import Redis
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Асинхронная обёртка над Redis для хранения состояния
    файлов, вопросов и ответов.
    """

    def __init__(self, url: str):
        self.url = url
        self.redis: Optional[Redis] = None

    async def connect(self) -> None:
        self.redis = Redis.from_url(
            self.url,
            decode_responses=True,
        )
        await self.redis.ping()
        logger.info("Redis успешно подключен")

    async def close(self) -> None:
        if self.redis:
            await self.redis.close()
            logger.info("Redis соединение закрыто")

    async def set_question(self, question_id: str, payload: dict) -> None:
        await self.redis.set(
            f"question:{question_id}",
            json.dumps(payload),
        )

    async def get_question(self, question_id: str) -> Optional[dict]:
        data = await self.redis.get(f"question:{question_id}")
        return json.loads(data) if data else None

    async def update_question(self, question_id: str, **fields) -> None:
        key = f"question:{question_id}"
        data = await self.redis.get(key)

        if not data:
            return

        payload = json.loads(data)
        payload.update(fields)

        await self.redis.set(key, json.dumps(payload))

    async def generate_file_id(self) -> str:
        """
        Генерирует простой инкрементный ID для файлов.
        """
        new_id = await self.redis.incr("file_id_counter")
        return str(new_id)

    async def generate_question_id(self) -> str:
        """
        Генерирует простой инкрементный ID для вопросов.
        """
        new_id = await self.redis.incr("question_id_counter")
        return str(new_id)
