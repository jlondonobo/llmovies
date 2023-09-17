from typing import Literal

import openai
from agents.constants import CATEGORIES
from instructor import OpenAISchema
from pydantic import Field


class QueryParams(OpenAISchema):
    "Correctly extracted media information"
    semantic_search: str = Field(
        ...,
        description="Topic to be used for semantic search. Must not include genre or media type.",
    )
    media: Literal["TV Show", "Movie", "ALL"] = Field(
        ...,
        description="Media type must be TV Show, Movie or ALL.",
    )
    genre: str | list[str] = Field(
        ...,
        description=f"MUST any combination of the following categories: {', '.join(CATEGORIES)}. If no genre is provided, return ALL.",
    )


def extract_query_params(input: str, chat_model: str) -> QueryParams:
    response = openai.ChatCompletion.create(
        model=chat_model,
        functions=[QueryParams.openai_schema],
        function_call={"name": QueryParams.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": "Extract media details from my requests. Only call functions if my input is related to movies or tv shows. If the user asks you anything different than movies or TV shows, respectfully stop the conversation.",
            },
            {"role": "user", "content": input},
        ],
    )

    return QueryParams.from_response(response)
