import json
import os
import re
from typing import Any, Literal

import openai
import streamlit as st
from dotenv import load_dotenv
from model import model
from pydantic import BaseModel, ValidationError
from streamlit.runtime.state import SessionStateProxy
from weaviate_client import client


class TopicInput(BaseModel):
    topic: str
    genre: str | None
    media: Literal["TV", "Movie", ""] | None


class LLMoviesOutputError(Exception):
    pass


load_dotenv()

openai.api_key = os.environ["OPENAI_KEY"]
chat_model = os.environ["OPENAI_CHAT_MODEL"]


categories = ["Action"]

PROMPT = f"""
Given a user input, return the topics, genre, and media type as a JSON object with the keys "topic", "genre", and "media".

You can use the following categories: {", ".join(categories)}.

You can use the following media types: TV, Movie.

You MUST not say anything after finishing.

You MUST only respond with a JSON object.

Your response will help filter some results, so don't say anything!

If the user asks you anything different than movies or TV shows, respectfully stop the conversation.
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
            raise LLMoviesOutputError("The response from the chatbot was not valid JSON.")
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

def search_movies(text: str, providers: list[str]) -> list[dict]:
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
    ]
    where_filter = {
        "path": ["providers"],
        "operator": "ContainsAny",
        "valueTextArray": providers,
    }
    res = (
        client.query.get("Movie", cols)
        .with_limit(3)
        .with_additional("distance")
        .with_near_vector(content={"vector": search_embedding})
        .with_where(where_filter)
        .do()["data"]["Get"]["Movie"]
    )
    return res

# -- APP --

FIRST_MESSAGE = "Hi there, it's your pal Tony here! What'd you like to watch?"

enqueue_first_message(st.session_state, PROMPT, FIRST_MESSAGE)
print_message_list(st.session_state.messages)

user_input = enqueue_user_message(st.session_state)
json_answer = handle_user_input(user_input, st.session_state)


if json_answer is not None:
    movies = search_movies(json_answer.topic, ["8"])
    st.write(movies)

