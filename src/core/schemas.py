from typing import Optional

from pydantic import BaseModel


class AskQuestionSchema(BaseModel):
    file_id: str
    question: str


class QuestionStatusResponse(BaseModel):
    question_id: str
    status: str
    answer: Optional[str] = None
    error: Optional[str] = None
