import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File, HTTPException

from src.core.logger import project_doc_for_log
from src.core.pipeline import RAGPipeline
from src.core.schemas import QuerySchema

logger = logging.getLogger(__name__)

health_router = APIRouter()

@health_router.get("/health", tags=["Health"])
async def health_check():
    """Проверка состояния сервиса."""
    logger.debug("Health check вызван")
    return {
        "status": "OK",
    }

generation_router = APIRouter(prefix="/generation", tags=["Generation"])

@generation_router.post("/generate")
async def generate(body: QuerySchema, req: Request):
    """
    Генерация ответа на пользовательский запрос.
    """
    logger.info("Запрос на генерацию ответа")

    try:
        pipeline: RAGPipeline = req.app.state.pipeline
        result = await pipeline.arun(query=body.query)

        req.state.rag_result = {
            "question": result.question,
            "documents": [
                project_doc_for_log(d, include_content=True)
                for d in result.documents
            ],
            "answer": result.answer,
        }

        logger.info("Ответ успешно сгенерирован")
        return {"answer": result.answer}

    except Exception:
        logger.exception("Ошибка при генерации ответа")
        raise HTTPException(500, "Ошибка генерации ответа")


ingestion_router = APIRouter(prefix="/ingestion", tags=["Ingestion"])

@ingestion_router.post("/ingest")
async def ingest(req: Request, file: UploadFile = File(...)):
    """
    Индексация DOCX-файла в RAG системе.
    """
    logger.info("Запрос на ingestion файла: %s", file.filename)

    suffix = Path(file.filename).suffix.lower()
    if suffix != ".docx":
        logger.warning("Попытка загрузки неподдерживаемого файла: %s", suffix)
        raise HTTPException(400, "Unsupported file type")

    tmp_path = None

    try:
        ingestion_service = req.app.state.ingest_service

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        result = await ingestion_service.ingest_file(tmp_path, file.filename)
        logger.info("Файл успешно проиндексирован: %s", file.filename)

        return {
            "status": "ok",
            **result,
        }

    except Exception:
        logger.exception("Ошибка при ingestion файла")
        raise HTTPException(500, "Ошибка ingestion файла")

    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)

retrieval_router = APIRouter(prefix="/retrieval", tags=["Retrieval"])

@retrieval_router.post("/retrieve")
async def retrieve(body: QuerySchema, req: Request):
    """
    Отбор релевантных чанков по пользовательскому запросу.
    """
    logger.info("Запрос на retrieval документов")

    try:
        retriever = req.app.state.retriever
        docs = await retriever.aretrieve(body.query)

        logger.info("Retrieval завершён, найдено документов: %d", len(docs))
        return {"documents": docs}

    except Exception:
        logger.exception("Ошибка при retrieval документов")
        raise HTTPException(500, "Ошибка retrieval документов")