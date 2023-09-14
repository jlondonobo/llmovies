import streamlit as st

from utils.model import model
from utils.weaviate_client import client

st.set_page_config(page_title="LLMovies", page_icon="🍿")


def search_movies(input: str) -> list[dict]:
    search_embedding = model.encode(input)
    cols = ["title", "description", "genres", "release_date", "vote_average", "trailer_url"]
    res =  (
        client.query.get("Movie", cols)
        .with_limit(15)
        .with_additional("distance")
        .with_near_vector(content={"vector": search_embedding})
        .do()["data"]["Get"]["Movie"]
    )
    return sorted(res, key=lambda x: x["vote_average"], reverse=True)

def reset_movies_state():
    st.session_state.movies = None
    st.session_state.page = 0

# Assuming page of 3 items
def get_item_location(page):
    start = page * 3
    end = page * 3 + 3
    return start, end 

# -- APP --

st.title("🎬 LLMovies")

st.subheader("Your go-to companion for movie nights")

search = st.text_input(
    "Hey there 👋🏻 it's ya boy! What are you looking for tonight",
    on_change=reset_movies_state,
)

if search == "":
    st.warning("You slacking? Let's get started by typing something in the search bar!")
    st.stop()
    
if st.session_state.movies is None:
    movies = search_movies(search)
    st.session_state.movies = movies
else:
    movies = st.session_state.movies 

num_movies = len(movies)
MAX_PAGES = min([num_movies // 3, 4])

col1, col2 = st.columns(2)

previous_is_disabled = st.session_state.page==0 
previous_button = col1.button("⬅️ Previous", disabled=previous_is_disabled, use_container_width=True)
if previous_button:
    st.session_state.page -= 1
    st.experimental_rerun()

next_is_disabled = st.session_state.page==MAX_PAGES
next_button = col2.button("Next ➡️", disabled=next_is_disabled, use_container_width=True)
if next_button:
    st.session_state.page = st.session_state.page + 1
    st.experimental_rerun()


def _prepare_youtube_url(video: str) -> str:
    return f"https://www.youtube.com/watch?v={video}"


start, end = get_item_location(st.session_state.page)
for movie in movies[start:end]:
    st.write(f"Title: {movie['title']}")
    st.write(f"Description: {movie['description']}")
    st.write(f"Genres: {movie['genres']}")
    st.write(f"Film score: {movie['vote_average']}")
    st.video(_prepare_youtube_url(movie["trailer_url"]))
    st.write("----")
