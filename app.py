import json
import logging
import os
from logging import basicConfig, getLogger
from typing import Any

import openai
import streamlit as st
import weaviate
from dotenv import load_dotenv

from utils import prompts
from utils.agents.input import extract_query_params
from utils.enums import Providers
from utils.exceptions import LLMoviesOutputError
from utils.model import model
from utils.weaviate_client import client

st.set_page_config(page_title="LLMovies", page_icon="🎬", layout="wide")

basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = getLogger(__name__)


def query_weaviate(
    text: str,
    providers: list[int],
    genres: str | list[str] | None,
    max_embed_recommendations: int,
    min_vote_count: int,
    input: str,
    weaviate_client: weaviate.Client,
) -> list[dict[str, Any]]:
    search_embedding = model.encode(text)
    cols = [
        "title",
        "description",
        "genres",
        "release_year",
        "vote_average",
        "vote_count",
        "trailer_url",
        "watch",
        "providers",
        "show_id",
        "runtime"
    ]
    operands = []
    providers_where = {
        "path": ["providers"],
        "operator": "ContainsAny",
        "valueInt": providers,
    }
    operands.append(providers_where)

    min_vote_count_where = {
        "path": ["vote_count"],
        "operator": "GreaterThan",
        "valueInt": min_vote_count,
    }
    operands.append(min_vote_count_where)

    if genres != "ALL":
        if isinstance(genres, str):
            genres = [genres]
        categories_where = {
            "path": ["genres"],
            "operator": "ContainsAny",
            "valueTextArray": genres,
        }
        operands.append(categories_where)

    where_filter = {"operator": "And", "operands": operands}
    logger.debug(f"Where filter: {json.dumps(where_filter)}")

    res = (
        weaviate_client.query.get("Movie", cols)
        .with_limit(max_embed_recommendations)
        .with_additional("distance")
        .with_near_vector(content={"vector": search_embedding})
        .with_where(where_filter)
        .with_generate(
            grouped_task=prompts.sort_movies_prompt.format(input),
            grouped_properties=[
                "title",
                "description",
                "release_date",
                "vote_average",
                "vote_count",
                "show_id",
            ],
        )
        .do()["data"]["Get"]["Movie"]
    )
    return res


def extract_titles_from(
    recommendations: str, possible_titles: list[str]
) -> list[str] | None:
    """Receives a string of titles from the chatbot and returns them as str."""

    def extract_titles(string: str) -> list[int]:
        try:
            split_titles = string.split("|")
            return [id for id in split_titles if id != ""]
        except ValueError:
            raise LLMoviesOutputError(
                "The response from the chatbot was not in the expected format."
            )

    def handle_valid_titles(titles: list[int]) -> None:
        if len(titles) == 0:
            raise LLMoviesOutputError(
                "Couldn't find any valid titles in the response from the chatbot."
            )

    titles = extract_titles(recommendations)

    valid_id_collection = [title for title in titles if title in possible_titles]
    handle_valid_titles(valid_id_collection)

    return valid_id_collection


def show_trailer(video: str | None):
    if video:
        st.video(f"https://www.youtube.com/watch?v={video}")
    else:
        st.write(prompts.no_trailer_message)


def get_provider_name(provider_id: str):
    return Providers(provider_id).name


def unsafe_html(html: str) -> st._DeltaGenerator:
    return st.markdown(html, unsafe_allow_html=True)


def format_runtime(runtime: int) -> str:
    hours = runtime // 60
    minutes = runtime % 60
    return f"{hours}h {minutes}m"


# -- APP --


def main():
    load_dotenv()
    CHAT_MODEL = os.environ["OPENAI_CHAT_MODEL"]
    N_MOVIES = 10
    MIN_VOTE_COUNT = 500
    # Initialize search_params and user_input
    search_params = None
    button_input = None
    user_input = None

    unsafe_html("<h1 style='text-align: center;'>🎬 LLMovies</h1>")
    unsafe_html(
        "<h3 style='text-align: center;'>Your go-to companion for movie nights</h3>"
    )
    with st.sidebar:
        # TODO: Remove default value in production
        openai_key = st.text_input(
            "Paste your OpenAI API key 🔑",
            type="password",
            value=os.getenv("OPENAI_KEY"),
        )

        if openai_key is None:
            st.warning("Hey! 🌟 Pop in your API key, and let's kick things off!")
        openai.api_key = openai_key

        available_services = st.multiselect(
            "Select your subscriptions 🍿",
            [p.value for p in Providers],
            format_func=get_provider_name,
            placeholder="What are you paying for?",
        )
        if available_services == []:
            st.warning("Ready to roll? Select your subscriptions first!")
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

        user_input = st.text_input("Send a message")

    if user_input != "" or button_input is not None:
        input = button_input if button_input is not None else user_input
        # add_user_message_to_history(input, st.session_state)
        try:
            search_params = extract_query_params(input, CHAT_MODEL)
        except openai.error.AuthenticationError:
            st.error(
                "Oops! It seems like your API key took a little detour. 🙃 Double-check and make sure it's the right one, will ya?"
            )
            st.stop()

    if search_params is not None:
        logger.debug(
            f"Response from params generation: {search_params.model_dump_json()}"
        )

        results_pool = query_weaviate(
            search_params.semantic_search,
            available_services,
            search_params.genre,
            N_MOVIES,
            MIN_VOTE_COUNT,
            input,
            client,
        )
        logger.debug(f"Movie pool: {json.dumps(results_pool)}")

        possible_titles = [result["title"] for result in results_pool]
        recommendations = results_pool[0]["_additional"]["generate"]["groupedResult"]
        logger.debug(f"Final recommendations: {recommendations}")

        try:
            recommended_titles = extract_titles_from(recommendations, possible_titles)

            if len(recommended_titles) != len(possible_titles):
                logger.debug(
                    f"Found {len(recommended_titles)} valid titles out of {len(possible_titles)} possible titles."
                )

            # Uses titles to sort results from Weaviate
            recommended_metadata = sorted(
                results_pool,
                key=lambda k: recommended_titles.index(k["title"])
                if k["title"] in recommended_titles
                else N_MOVIES + 1,
            )

            # Renders final recommendations
            cols = st.columns(3)

            # TODO: Move this to css file
            unsafe_html(
                """
                <style>
                .list-inline {
                    list-style: none;
                    margin-left: 0;
                    padding-left: 0;
                }

                .list-inline > li {
                    display: inline-block;
                    margin-left: 0;
                    padding-left: 0;
                    color: #737373;
                    margin-right: 0.4em;
                }
                
                .movie-title {
                    padding-bottom: 0px;
                }
                
                .list-inline li:nth-child(2)::before {
                    content: "·";  
                    margin-right: 0.4em;
                }
                .list-inline li:nth-child(2) {
                    line-height: 1.5; 
                }
                </style>
                """
            )

            for idx, movie in enumerate(recommended_metadata[:3]):
                with cols[idx]:
                    unsafe_html(f"<h3 class='movie-title'>{movie['title']} </h3>")
                    unsafe_html(
                        f"""
                        <ul class="list-inline">
                        <li>{movie['release_year']}</li>
                        <li>{format_runtime(movie['runtime'])}</li>
                        </ul>
                        """
                    )

                    # unsafe_html(f"<a href={movie['watch']}> 👀 </a>")
                    st.markdown(f"*{movie['description']}*")
                    st.write(f"Genres: {movie['genres']}")
                    st.write(f"Film score: {movie['vote_average']}")
                    st.write(f"Cosine distance: {movie['_additional']['distance']}")
                    show_trailer(movie["trailer_url"])
                    st.write("----")

        except LLMoviesOutputError:
            st.write(
                "I wasn't able to find any movies for you. Try modifying your query."
            )


if __name__ == "__main__":
    main()
