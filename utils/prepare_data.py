# Read netflix movies
import datasets
from model import model
from tqdm import tqdm
from weaviate_client import client

DATA_SOURCE = "data/final_movies.parquet"
movies = datasets.DatasetDict.from_parquet(DATA_SOURCE)


def format_description(row: dict[str, str | list]) -> str:
    title = row["title"]
    description = row["overview"]
    genres = row["genres_list"]
    return f"Title: {title}\nDescription: {description}\nGenres: {genres}"

processed_samples = []
for idx, val in tqdm(enumerate(movies)):
    full_description = format_description(val)
    embedding = model.encode(full_description)
    processed_samples.append(
        {
            "show_id": val["id"],
            "title": val["title"],
            "description": val["overview"],
            "release_date": val["release_date"],
            "genres": val["genres_list"],
            "trailer_url": val["trailer"],
            "watch": val["provider_url"],
            "vote_average": val["vote_average"],
            "full_description": full_description,
            "embedding": embedding,
        }
    )

features = datasets.Features(
    {
        "show_id": datasets.Value(dtype="string"),
        "title": datasets.Value(dtype="string"),
        "description": datasets.Value(dtype="string"),
        "release_date": datasets.Value(dtype="string"),
        "genres": datasets.Value(dtype="string"),
        "full_description": datasets.Value(dtype="string"),
        "trailer_url": datasets.Value(dtype="string"),
        "watch": datasets.Value(dtype="string"),
        "vote_average": datasets.Value(dtype="string"),
        "embedding": datasets.Sequence(
            feature=datasets.Value(dtype="float32"), length=384
        ),
    }
)
movies_dataset = datasets.Dataset.from_list(processed_samples, features=features)


class_definition = {
    "class": "Movie",
    "vectorIndexConfig": {
        "distance": "cosine",
    },
    "moduleConfig": {"generative-openai": {}},
    "properties": [
        {"name": "show_id", "dataType": ["text"]},
        {"name": "title", "dataType": ["text"]},
        {"name": "description", "dataType": ["text"]},
        {"name": "release_date", "dataType": ["text"]},
        {"name": "genres", "dataType": ["text"]},
        {"name": "trailer_url", "dataType": ["text"]},
        {"name": "watch", "dataType": ["text"]},
        {"name": "vote_average", "dataType": ["text"]},
        {"name": "full_description", "dataType": ["text"]},
    ],
}

client.schema.delete_class("Movie")

if not client.schema.exists("Movie"):
    client.schema.create_class(class_definition)

client.batch.configure(batch_size=100)
with client.batch as batch:
    for mov in movies_dataset:
        properties = {
            "title": mov["title"],
            "description": mov["description"],
            "full_description": mov["full_description"],
            "release_date": mov["release_date"],
            "genres": mov["genres"],
            "trailer_url": mov["trailer_url"],
            "watch": mov["watch"],
            "vote_average": mov["vote_average"],
        }
        batch.add_data_object(properties, class_name="Movie", vector=mov["embedding"])
