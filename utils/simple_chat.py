import json
import os
import re
from typing import Literal

import openai
import streamlit as st
from dotenv import load_dotenv
from model import model
from pydantic import BaseModel, ValidationError
from streamlit.runtime.state import SessionStateProxy
from weaviate_client import client

MAX_EMBED_RECOMMENDATIONS = 20


class TopicInput(BaseModel):
    topic: str
    genre: str | None
    media: Literal["TV", "Movie", ""] | None | str


class RecommendationList(BaseModel):
    ids: list[int]


class LLMoviesOutputError(Exception):
    pass


load_dotenv()

openai.api_key = os.environ["OPENAI_KEY"]
chat_model = os.environ["OPENAI_CHAT_MODEL"]


categories = ["Action"]

INPUT_PROMPT = f"""
Given a user input, return the topics, genre, and media type as a JSON object with the keys "topic", "genre", and "media".

You can use the following categories: {", ".join(categories)}.

You can use the following media types: TV, Movie.

You MUST not say anything after finishing.

You MUST only respond with a JSON object.

Your response will help filter some results, so don't say anything!

If the user asks you anything different than movies or TV shows, respectfully stop the conversation.
"""

OUTPUT_PROMPT = """
You are an expert movie recommender system. Your task is to return at most 3 movies from the list of passed movies. Return only the most affine to the user's prompt. If no movie is related to the user's prompt ask him to try again.

You will only respond with a list of the sorted ids separated by commas, and nothing else. You must not add anything else to your answer
"""


def generate_response(input: str, history: list[dict[str, str]]) -> str:
    """Returns the response from the chatbot."""
    if input is None:
        st.warning("Please enter a message.")
        st.stop()
    messages = history + [{"role": "user", "content": input}]

    response = openai.ChatCompletion.create(
        model=chat_model,
        messages=messages,
    )
    return response["choices"][0]["message"]["content"]


def parse_response(response: str) -> TopicInput:
    """Make sure the response is valid JSON and can be parsed into a TopicInput."""
    try:
        as_json = json.loads(response)
        topic_input = TopicInput(**as_json)
        return topic_input
    except json.decoder.JSONDecodeError:
        try:
            as_json = extract_json_from_string(response)
            return TopicInput(**as_json)
        except json.decoder.JSONDecodeError:
            raise LLMoviesOutputError(
                "The response from the chatbot was not valid JSON."
            )
    except ValidationError as e:
        print(e.errors())
        raise LLMoviesOutputError


def extract_json_from_string(string: str) -> dict:
    pattern = r"(?<=```json\s)(\{[\s\S]*?\})(?=\s*```)"
    match = re.search(pattern, string)
    json_string = match.group(1)
    return json.loads(json_string)


def enqueue_first_message(
    state: SessionStateProxy, system_message: str, first_message: str
) -> None:
    if "messages" not in state.keys():
        state.messages = [
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": first_message},
        ]


def is_valid_json(string: str) -> bool:
    try:
        json.loads(string)
    except ValueError:
        return False
    return True


def print_message_list(messages: list[dict[str, str]]) -> None:
    for message in messages:
        is_json = is_valid_json(message["content"])
        is_system = message["role"] == "system"
        if not (is_json or is_system):
            with st.chat_message(message["role"]):
                st.write(message["content"])


def enqueue_user_message(state: SessionStateProxy) -> str:
    if user_input := st.chat_input("Send a message"):
        message = {"role": "user", "content": user_input}
        state.messages.append(message)
        with st.chat_message("user"):
            st.write(user_input)
    return user_input


def handle_user_input(user_input: str, state: SessionStateProxy) -> None | TopicInput:
    is_last_message_from_user = state.messages[-1]["role"] == "user"
    if is_last_message_from_user:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(user_input, st.session_state.messages)
                try:
                    return parse_response(response)
                except LLMoviesOutputError:
                    st.write(response)

        message = {"role": "assistant", "content": response}
        state.messages.append(message)


def search_movies(
    text: str,
    providers: list[str],
    max_embed_recommendations: int,
) -> list[dict]:
    search_embedding = model.encode(text)
    cols = [
        "title",
        "description",
        "genres",
        "release_date",
        "vote_average",
        "trailer_url",
        "watch",
        "providers",
        "show_id",
    ]
    where_filter = {
        "path": ["providers"],
        "operator": "ContainsAny",
        "valueTextArray": providers,
    }
    res = (
        client.query.get("Movie", cols)
        .with_limit(max_embed_recommendations)
        .with_additional("distance")
        .with_near_vector(content={"vector": search_embedding})
        .with_where(where_filter)
        .do()["data"]["Get"]["Movie"]
    )
    return res


def get_final_recommendations(movies_list: dict, input: str) -> RecommendationList:
    json_input = movies_list | {"user_prompt": input}
    response = openai.ChatCompletion.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": OUTPUT_PROMPT},
            {"role": "user", "content": json.dumps(json_input)},
        ],
    )
    return response["choices"][0]["message"]["content"]


def format_movie_list(movies: list[dict]):
    movies_list = {
        "list": [
            {
                "id": movie["show_id"],
                "title": movie["title"],
                "description": movie["description"],
                "genres": movie["genres"],
            }
            for movie in movies
        ]
    }
    movies_ids = [int(movie["id"]) for movie in movies_list["list"]]
    return movies_list, movies_ids


def parse_final_recommendations(
    response: str, movies_ids: list[int]
) -> list[int] | None:
    try:
        ids = [int(id) for id in response.split(", ") if id != ""]
        recommendation_list = RecommendationList(ids=ids)
    except ValueError:
        raise LLMoviesOutputError(
            "The response from the chatbot was not in the expected format."
        )
    if not all(id in movies_ids for id in ids):
        raise LLMoviesOutputError(
            "The response from the chatbot contained invalid ids."
        )
    else:
        return recommendation_list.ids


def _prepare_youtube_url(video: str) -> str:
    # TODO: do something if video is None
    return f"https://www.youtube.com/watch?v={video}"


# -- APP --

FIRST_MESSAGE = "Hi there, it's your pal Tony here! What'd you like to watch?"

enqueue_first_message(st.session_state, INPUT_PROMPT, FIRST_MESSAGE)
print_message_list(st.session_state.messages)

user_input = enqueue_user_message(st.session_state)
json_answer = handle_user_input(user_input, st.session_state)


if json_answer is not None:
    movies = search_movies(json_answer.topic, ["8"], MAX_EMBED_RECOMMENDATIONS)
    movies_list, movies_ids = format_movie_list(movies)
    final_recommendations = get_final_recommendations(movies_list, user_input)
    with st.chat_message("assistant"):
        try:
            recommendation_ids = parse_final_recommendations(
                final_recommendations, movies_ids
            )

            final_movies = [
                movie for movie in movies if int(movie["show_id"]) in recommendation_ids
            ]

            for movie in final_movies:
                st.markdown(
                    f"<h2>{movie['title']}</h2><p><a href={movie['watch']}>View</a></p>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"*{movie['description']}*")
                st.write(f"Genres: {movie['genres']}")
                st.write(f"Release date: {movie['release_date']}")
                st.write(f"Film score: {movie['vote_average']}")
                st.write(f"Cosine distance: {movie['_additional']['distance']}")
                st.video(_prepare_youtube_url(movie["trailer_url"]))
                st.write("----")

        except LLMoviesOutputError:
            st.write(final_recommendations)
