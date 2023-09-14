# Read netflix movies
import datasets
from model import model
from tqdm import tqdm
from weaviate_client import client

DATA_SOURCE = "data/netflix_titles.csv"
movies = datasets.DatasetDict.from_csv(DATA_SOURCE)


def format_description(row: dict[str, str | list]) -> str:
    title = row["title"]
    description = row["description"]
    genres = ", ".join(row["listed_in"])
    return f"Title: {title}\nDescription: {description}\nGenres: {genres}"


processed_samples = []
for idx, val in tqdm(enumerate(movies)):
    full_description = format_description(val)
    embedding = model.encode(full_description)
    processed_samples.append(
        {
            "show_id": val["show_id"],
            "title": val["title"],
            "description": val["description"],
            "release_year": val["release_year"],
            "listed_in": val["listed_in"],
            "rating": val["rating"],
            "full_description": full_description,
            "embedding": embedding,
        }
    )

features = datasets.Features(
    {
        "show_id": datasets.Value(dtype="string"),
        "title": datasets.Value(dtype="string"),
        "description": datasets.Value(dtype="string"),
        "release_year": datasets.Value(dtype="int32"),
        "listed_in": datasets.Value(dtype="string"),
        "rating": datasets.Value(dtype="string"),
        "full_description": datasets.Value(dtype="string"),
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
        {"name": "release_year", "dataType": ["int"]},
        {"name": "listed_in", "dataType": ["text"]},
        {"name": "rating", "dataType": ["text"]},
        {"name": "full_description", "dataType": ["text"]},
    ],
}

# client.schema.delete_class("Movie")

if not client.schema.exists("Movie"):
    client.schema.create_class(class_definition)

client.batch.configure(batch_size=100)
with client.batch as batch:
    for mov in movies_dataset:
        properties = {
            "title": mov["title"],
            "description": mov["description"],
            "full_description": mov["full_description"],
            "release_year": mov["release_year"],
            "listed_in": mov["listed_in"],
            "rating": mov["rating"],
        }
        batch.add_data_object(properties, class_name="Movie", vector=mov["embedding"])
