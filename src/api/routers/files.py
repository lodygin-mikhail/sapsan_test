import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File, HTTPException

logger = logging.getLogger(__name__)

files_router = APIRouter(prefix="/files", tags=["Files"])

@files_router.post("")
async def upload_file(req: Request, file: UploadFile = File(...)):
    """
    Загружает файл и запускает ingestion.
    Возвращает file_id.
    """
    logger.info("Загрузка файла: %s", file.filename)

    suffix = Path(file.filename).suffix.lower()
    if suffix != ".docx":
        raise HTTPException(400, "Unsupported file type")

    file_id = await req.app.state.redis.generate_file_id()
    tmp_path = None

    try:
        ingestion_service = req.app.state.ingest_service

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        await ingestion_service.ingest_file(tmp_path, file.filename)

        return {
            "file_id": file_id,
            "status": "uploaded",
        }

    except Exception:
        logger.exception("Ошибка при загрузке файла")
        raise HTTPException(500, "Ошибка загрузки файла")

    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)