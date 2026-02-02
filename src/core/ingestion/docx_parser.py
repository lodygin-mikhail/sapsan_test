import logging
import re
from typing import List

from docx import Document as DocxDocument
from langchain_core.documents import Document


logger = logging.getLogger(__name__)

class DocxParser:
    """
    Парсер DOCX файлов для RAG систем.

    Разбивает документ на текстовые блоки и таблицы,
    формируя Document объекты с метаданными.
    """

    def __init__(self, docx_path: str):
        logger.info("Инициализация DocxParser: %s", docx_path)
        try:
            self.doc = DocxDocument(docx_path)
        except Exception:
            logger.exception("Ошибка открытия DOCX файла")
            raise

        self.current_section = None

    def parse(self) -> List[Document]:
        """
        Парсит DOCX документ и возвращает список Document объектов.
        """
        logger.info("Начало парсинга DOCX документа")
        documents = []

        try:
            full_text = ""
            for para in self.doc.paragraphs:
                full_text += para.text + "\n"

            full_text = full_text.replace("\\n", "\n")
            lines = full_text.split("\n")

            current_content = []
            idx = 0

            while idx < len(lines):
                line = lines[idx].strip()

                if not line:
                    idx += 1
                    continue

                if self._is_table_start(lines, idx):
                    if current_content:
                        documents.append(
                            self._create_document(
                                text="\n".join(current_content),
                                chunk_type="paragraph",
                            )
                        )
                        current_content = []

                    table_md, new_idx = self._extract_table(lines, idx)
                    documents.append(
                        self._create_document(
                            text=table_md,
                            chunk_type="table",
                            is_atomic=True,
                        )
                    )
                    idx = new_idx
                    continue

                if self._is_main_section(line):
                    if current_content:
                        documents.append(
                            self._create_document(
                                text="\n".join(current_content),
                                chunk_type="paragraph",
                            )
                        )
                        current_content = []

                    self.current_section = line
                    logger.debug("Новая секция: %s", line)

                current_content.append(line)
                idx += 1

            if current_content:
                documents.append(
                    self._create_document(
                        text="\n".join(current_content),
                        chunk_type="paragraph",
                    )
                )

            logger.info("Парсинг завершён, документов: %d", len(documents))
            return documents

        except Exception:
            logger.exception("Ошибка при парсинге DOCX документа")
            raise

    def _is_main_section(self, line: str) -> bool:
        """Определяет главный заголовок раздела"""
        if not line:
            return False

        # Только главные разделы: "1. ЗАГОЛОВОК"
        if re.match(r'^(\d+)\.\s+([А-ЯЁ][А-ЯЁ\s]+)$', line):
            return True

        # Приложения: "Приложение №1"
        if re.match(r'^(Приложение\s*№\s*\d+)', line, re.IGNORECASE):
            return True

        # Markdown заголовки: "# Заголовок" или "## Заголовок"
        if re.match(r'^(#{1,2})\s+(.+)$', line):
            return True

        return False

    def _is_table_start(self, lines: List[str], idx: int) -> bool:
        """Проверяет начало markdown-таблицы"""
        if idx >= len(lines) or idx + 1 >= len(lines):
            return False

        line1 = lines[idx].strip()
        line2 = lines[idx + 1].strip()

        if '|' not in line1 or '|' not in line2:
            return False

        # Вторая строка должна быть разделителем (|---|---|)
        if re.match(r'^\|[\s\-:|]+\|$', line2):
            return True

        return False

    def _extract_table(self, lines: List[str], start_idx: int):
        """Извлекает таблицу, возвращает (table_text, next_idx)"""
        table_lines = [lines[start_idx].strip()]
        idx = start_idx + 1

        while idx < len(lines):
            line = lines[idx].strip()
            if '|' in line:
                table_lines.append(line)
                idx += 1
            else:
                break

        return '\n'.join(table_lines), idx

    def _create_document(
            self,
            text: str,
            chunk_type: str,
            is_atomic: bool = False
    ) -> Document:
        """Создаёт Document объект с метаданными"""
        metadata = {
            "chunk_type": chunk_type,
            "section": self.current_section,
            "subsection": None,  # Для совместимости с IngestionService
        }

        if is_atomic:
            metadata["is_atomic"] = True

        return Document(
            page_content=text,
            metadata=metadata
        )