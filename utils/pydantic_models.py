from typing import Literal

from pydantic import BaseModel


class TopicInput(BaseModel):
    topic: str
    genre: str | None
    media: Literal["TV", "Movie", ""] | None | str
