import asyncio
import logging

from fastapi import APIRouter, Request, HTTPException

from src.core.pipeline import RAGPipeline
from src.core.schemas import AskQuestionSchema, QuestionStatusResponse

logger = logging.getLogger(__name__)

questions_router = APIRouter(prefix="/questions", tags=["Questions"])


@questions_router.post("")
async def ask_question(body: AskQuestionSchema, req: Request):
    """
    Принимает вопрос по файлу и возвращает question_id.
    Обработка выполняется асинхронно.
    """
    redis = req.app.state.redis
    question_id = await redis.generate_question_id()

    payload = {
        "question_id": question_id,
        "file_id": body.file_id,
        "question": body.question,
        "status": "processing",
        "answer": None,
    }

    await redis.set_question(question_id, payload)

    async def process_question():
        try:
            pipeline: RAGPipeline = req.app.state.pipeline
            result = await pipeline.arun(query=body.question)

            await redis.update_question(
                question_id,
                status="done",
                answer=result.answer,
            )

        except Exception as e:
            logger.exception("Ошибка обработки вопроса %s", question_id)
            await redis.update_question(
                question_id,
                status="error",
                error=str(e),
            )

    # запускаем в фоне
    asyncio.create_task(process_question())

    return {
        "question_id": question_id,
        "status": "processing",
    }


@questions_router.get("/{question_id}", response_model=QuestionStatusResponse)
async def get_question_status(question_id: str, req: Request):
    """
    Возвращает статус обработки вопроса или готовый ответ.
    """
    redis = req.app.state.redis
    data = await redis.get_question(question_id)

    if not data:
        raise HTTPException(404, "Question not found")

    return data
