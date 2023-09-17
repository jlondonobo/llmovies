import numpy as np
import pandas as pd
import weaviate
from model import model
from tqdm import tqdm
from weaviate_client import client

tqdm.pandas(desc="Processing embeddings")

CLASS_DEFINITION = {
    "class": "Movie",
    "vectorIndexConfig": {
        "distance": "cosine",
    },
    "moduleConfig": {"generative-openai": {}},
    "properties": [
        {"name": "show_id", "dataType": ["int"]},
        {"name": "title", "dataType": ["text"]},
        {"name": "description", "dataType": ["text"]},
        {"name": "release_date", "dataType": ["text"]},
        {"name": "genres", "dataType": ["text[]"]},
        {"name": "trailer_url", "dataType": ["text"]},
        {"name": "watch", "dataType": ["text"]},
        {"name": "providers", "dataType": ["int[]"]},
        {"name": "vote_average", "dataType": ["number"]},
        {"name": "vote_count", "dataType": ["int"]},
        {"name": "full_description", "dataType": ["text"]},
        {"name": "imdb_vote_average", "dataType": ["number"]},
        {"name": "imdb_vote_count", "dataType": ["int"]},
    ],
}


def download_imdb_ratings() -> pd.DataFrame:
    url = "https://datasets.imdbws.com/title.ratings.tsv.gz"
    return pd.read_csv(url, sep="\t")


def add_imdb_ratings(movies: pd.DataFrame, imdb_ratings: pd.DataFrame) -> pd.DataFrame:
    # Some movies have no IMDb id (None), so we need m:1
    imdb_formatted = imdb_ratings.rename(
        columns={
            "tconst": "imdb_id",
            "averageRating": "imdb_vote_average",
            "numVotes": "imdb_vote_count",
        }
    )

    merged = movies.merge(
        imdb_formatted,
        on="imdb_id",
        how="left",
        validate="m:1",
    )
    return merged


def read_movies(source: str) -> pd.DataFrame:
    res = pd.read_parquet(source)
    return res.assign(
        providers=lambda df: df["providers"].apply(np.ndarray.tolist),
        genres_list=lambda df: df["genres_list"].str.split(", ")
    )


def format_description(row: dict[str, str | list]) -> str:
    return row["overview"]


def add_embeddings(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(
        full_description=lambda df: df.apply(format_description, axis=1),
        embedding=lambda df: df["full_description"].progress_apply(model.encode),
    )

def parse_null_float(val: float) -> float | None:
    if np.isnan(val):
        return None
    return val

def parse_null_int(val: int) -> int | None:
    if np.isnan(val):
        return None
    return int(val)

def save_to_weaviate(data: pd.DataFrame, client: weaviate.Client) -> None:
    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for idx, row in data.iterrows():
            properties = {
                "show_id": row["id"],
                "title": row["title"],
                "description": row["overview"],
                "full_description": row["full_description"],
                "release_date": row["release_date"],
                "genres": row["genres_list"],
                "trailer_url": row["trailer"],
                "watch": row["provider_url"],
                "providers": row["providers"],
                "vote_average": parse_null_float(row["vote_average"]),
                "vote_count": row["vote_count"],
                "imdb_vote_average": parse_null_float(row["imdb_vote_average"]),
                "imdb_vote_count": parse_null_int(row["imdb_vote_count"]),
            }
            batch.add_data_object(
                properties, class_name="Movie", vector=row["embedding"]
            )


def main():
    DATA_SOURCE = "data/final_movies.parquet"
    movies = read_movies(DATA_SOURCE)
    imdb_ratings = download_imdb_ratings()
    moviews_with_imbd_ratings = add_imdb_ratings(movies, imdb_ratings)
    movies_with_embeddings = add_embeddings(moviews_with_imbd_ratings)

    if client.schema.exists("Movie"):
        client.schema.delete_class("Movie")
    client.schema.create_class(CLASS_DEFINITION)
    save_to_weaviate(movies_with_embeddings, client)


if __name__ == "__main__":
    main()
