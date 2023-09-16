import json
import os

import openai
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel


class RecommendationList(BaseModel):
    ids: list[int]


class LLMoviesOutputError(Exception):
    pass


load_dotenv()

openai.api_key = os.environ["OPENAI_KEY"]
chat_model = os.environ["OPENAI_CHAT_MODEL"]

PROMPT = """
You are an expert movie recommender system. Your task is to return at most 3 movies from the list of passed movies. Return only the most affine to the user's prompt. If no movie is related to the user's prompt ask him to try again.

You will only respond with a list of the sorted ids separated by commas, and nothing else. You must not add anything else to your answer
"""


def calculate_movie_list():
    movies_list = json.load(open("data/test_movies_list.json", "r"))
    movies_ids = [movie["id"] for movie in movies_list["list"]]
    return movies_list, movies_ids


def generate_response(
    movies_list: dict, input: str, messages: list[dict[str, str]]
) -> RecommendationList:
    json_input = movies_list | {"user_prompt": input}
    response = openai.ChatCompletion.create(
        model=chat_model,
        messages=messages + [{"role": "user", "content": json.dumps(json_input)}],
    )
    return response["choices"][0]["message"]["content"]


def parse_response(response: str, movies_ids: list[int]) -> RecommendationList:
    try:
        ids = response.strip().split(",")
        ids = [int(id) for id in ids if id != ""]
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
        return recommendation_list


if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "system", "content": PROMPT},
        {
            "role": "assistant",
            "content": "Add the user prompt here, We're retrieving the movies list obtained previously in the backyardigans",
        },
    ]

# Print messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

if user_input := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    movies_list, movies_ids = calculate_movie_list()
    with st.chat_message("user"):
        st.write(user_input)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(
                movies_list, user_input, st.session_state.messages
            )
            try:
                parsed = parse_response(response, movies_ids)
                st.image(
                    "https://t3.ftcdn.net/jpg/01/44/89/98/360_F_144899855_tqFVpAifgNGq7Uti1XoANq0mvsg0518o.jpg"
                )
            except LLMoviesOutputError:
                st.write(response)

    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
