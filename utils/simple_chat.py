import json
import logging
import os
import re
from logging import basicConfig, getLogger
from typing import Any

import openai
import prompts
import streamlit as st
import weaviate
from dotenv import load_dotenv
from enums import Providers
from exceptions import LLMoviesOutputError
from model import model
from pydantic import ValidationError
from pydantic_models import TopicInput
from streamlit.runtime.state import SessionStateProxy
from weaviate_client import client

basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = getLogger(__name__)


def generate_response(
    input: str, history: list[dict[str, str]], chat_model: str
) -> str:
    """Returns the response from the chatbot."""
    if input is None:
        st.warning("Please enter a message.")
        st.stop()
    messages = history + [{"role": "user", "content": input}]

    try:
        response = openai.ChatCompletion.create(
            model=chat_model,
            messages=messages,
        )
        return response["choices"][0]["message"]["content"]
    except openai.error.AuthenticationError:
        st.error(
            "Oops! It seems like your API key took a little detour. 🙃 Double-check and make sure it's the right one, will ya?"
        )
        st.stop()


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
        except AttributeError:
            raise LLMoviesOutputError(
                "The response from the chatbot was not valid JSON."
            )
    except ValidationError as e:
        print(e.errors())
        raise LLMoviesOutputError


def extract_json_from_string(string: str) -> dict:
    pattern = r"\{[\s\S]*?\}"
    match = re.search(pattern, string)
    json_string = match.group(1)
    return json.loads(json_string)


def chatbot_setup(
    system_message: str,
    initial_assistant_message: str,
    state: SessionStateProxy,
) -> None:
    if "messages" not in state.keys():
        state.messages = [
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": initial_assistant_message},
        ]


def is_valid_json(string: str) -> bool:
    try:
        json.loads(string)
    except ValueError:
        return False
    return True


def render_chat_history(messages: list[dict[str, str]]) -> None:
    for message in messages:
        is_json = is_valid_json(message["content"])
        is_system = message["role"] == "system"
        if not (is_json or is_system):
            with st.chat_message(message["role"]):
                st.write(message["content"])


def add_user_message_to_history(user_input: str, state: SessionStateProxy) -> str:
    message = {"role": "user", "content": user_input}
    state.messages.append(message)
    with st.chat_message("user"):
        st.write(user_input)
    return user_input


def try_extract_search_params(
    user_input: str, state: SessionStateProxy, chat_model: str
) -> None | TopicInput:
    """Tries to extract search parameters from input, if not possible renders answer."""
    is_last_message_from_user = state.messages[-1]["role"] == "user"
    if is_last_message_from_user:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(
                    user_input, st.session_state.messages, chat_model
                )
                try:
                    return parse_response(response)
                except LLMoviesOutputError:
                    st.write(response)

        message = {"role": "assistant", "content": response}
        state.messages.append(message)


def query_weaviate(
    text: str,
    providers: list[str],
    genres: str | list[str] | None,
    max_embed_recommendations: int,
    min_vote_count: int,
    weaviate_client: weaviate.Client,
) -> list[dict[str, Any]]:
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

    operands = []
    providers_where = {
        "path": ["providers"],
        "operator": "ContainsAny",
        "valueTextArray": providers,
    }
    operands.append(providers_where)
    min_vote_count_where = {
        "path": ["vote_count"],
        "operator": "GreaterThan",
        "valueInt": min_vote_count,
    }
    operands.append(min_vote_count_where)
    if genres is not None:
        if isinstance(genres, str):
            genres = [genres]
        categories_where = {
            "path": ["genres"],
            "operator": "ContainsAny",
            "valueTextArray": genres,
        }
        operands.append(categories_where)

    where_filter = {"operator": "And", "operands": operands}

    res = (
        weaviate_client.query.get("Movie", cols)
        .with_limit(max_embed_recommendations)
        .with_additional("distance")
        .with_near_vector(content={"vector": search_embedding})
        .with_where(where_filter)
        .do()["data"]["Get"]["Movie"]
    )
    return res


def llm_generate_recommendation(
    user_message: str,
    formatted_results: dict[str, list[dict[str, Any]]],
    chat_model: str,
) -> str:
    """Receives formatted input & movies. Outputs final recommendations."""
    json_input = formatted_results | {"user_prompt": user_message}
    rendered_input = json.dumps(json_input)

    response = openai.ChatCompletion.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": prompts.final_recommendations_system},
            {"role": "user", "content": rendered_input},
        ],
    )
    return response["choices"][0]["message"]["content"]


def format_query_results(
    results: list[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], list[int]]:
    """Returns results as expected by LLM and a list of ids."""
    formatted_collection = []
    ids_collection = []
    for result in results:
        formatted = {
            "id": result["show_id"],
            "title": result["title"],
            "description": result["description"],
            "genres": result["genres"],
        }
        formatted_collection.append(formatted)
        ids_collection.append(result["show_id"])

    final_collection = {"list": formatted_collection}
    return final_collection, ids_collection


def extract_ids_from(recommendations: str, possible_ids: list[int]) -> list[int] | None:
    """Receives a string of ids from the chatbot and returns them as int if valid."""

    def extract_ids_as_int(string: str) -> list[int]:
        try:
            split_ids = string.split(", ")
            return [int(id) for id in split_ids if id != ""]
        except ValueError:
            raise LLMoviesOutputError(
                "The response from the chatbot was not in the expected format."
            )

    def handle_valid_ids(ids: list[int]) -> None:
        if len(ids) == 0:
            raise LLMoviesOutputError(
                "Couldn't find any valid ids in the response from the chatbot."
            )

    ids_as_int = extract_ids_as_int(recommendations)

    valid_id_collection = [id for id in ids_as_int if id in possible_ids]
    handle_valid_ids(valid_id_collection)

    return valid_id_collection


def _prepare_youtube_url(video: str) -> str:
    # TODO: do something if video is None
    return f"https://www.youtube.com/watch?v={video}"


def get_provider_name(provider_id: str):
    return Providers(provider_id).name


# -- APP --
load_dotenv()

# -- Parameters -- #


def main():
    CHAT_MODEL = os.environ["OPENAI_CHAT_MODEL"]
    INITIAL_ASSISTANT_MESSAGE = (
        "Hi there, it's your pal Tony here! What'd you like to watch?"
    )
    N_MOVIES = 20
    MIN_VOTE_COUNT = 30
    # Initialize search_params and user_input
    search_params = None
    button_input = None
    user_input = None

    st.title("🎬 LLMovies")
    st.subheader("Your go-to companion for movie nights")
    with st.sidebar:
        # TODO: Remove default value in production
        openai_key = st.text_input(
            "Your OpenAI API key 🔑", type="password", value=os.getenv("OPENAI_KEY")
        )

        if openai_key is None:
            st.warning("Hey! 🌟 Pop in your API key, and let's kick things off!")
        openai.api_key = openai_key

        available_services = st.multiselect(
            "Your subscriptions 🍿",
            [p.value for p in Providers],
            format_func=get_provider_name,
            placeholder="What are you paying for?",
        )
        if available_services == []:
            st.warning("Next up, tap on your movie subscriptions! 🎬 Ready to roll?")
            st.stop()

        st.subheader("Try me out! 🤖")
        q1 = "I'd like to watch a movie about friendship."
        if st.button(q1):
            button_input = q1

        q2 = "Can you recommend a comedy located in Italy?"
        if st.button(q2):
            button_input = q2

        q3 = "Do you know any action movies with lots of explosions?"
        if st.button(q3):
            button_input = q3

    chatbot_setup(prompts.setup_system, INITIAL_ASSISTANT_MESSAGE, st.session_state)
    render_chat_history(st.session_state.messages)

    user_input = st.chat_input("Send a message")

    if user_input is not None or button_input is not None:
        input = button_input if button_input is not None else user_input
        user_message = add_user_message_to_history(input, st.session_state)

        search_params = try_extract_search_params(
            user_message, st.session_state, CHAT_MODEL
        )

    if search_params is not None:
        logger.debug(
            f"Response from params generation: {search_params.model_dump_json()}"
        )

        results_pool = query_weaviate(
            search_params.topic,
            available_services,
            search_params.genres,
            N_MOVIES,
            MIN_VOTE_COUNT,
            client,
        )
        logger.debug(f"Movie pool: {json.dumps(results_pool)}")

        formatted_weaviate_results, possible_ids = format_query_results(results_pool)
        recommendations = llm_generate_recommendation(
            user_message, formatted_weaviate_results, CHAT_MODEL
        )
        logger.debug(f"Final recommendations: {recommendations}")

        with st.chat_message("assistant"):
            try:
                # Extracts ids (as int) from LLM response
                recommended_ids = extract_ids_from(recommendations, possible_ids)

                # Uses extracted ids to filter results from Weaviate
                recommended_metadata = [
                    result
                    for result in results_pool
                    if result["show_id"] in recommended_ids
                ]

                # Renders final recommendations
                for movie in recommended_metadata:
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
                st.write(
                    "I wasn't able to find any movies for you. Try modifying your query."
                )


if __name__ == "__main__":
    main()
