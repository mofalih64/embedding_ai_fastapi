from pydantic import BaseModel
from pydantic import UUID4




class returnResponse(BaseModel):
    answer: str


class question(BaseModel):
    question: str
