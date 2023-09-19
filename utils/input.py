from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.schema import Document
from langchain.vectorstores import Weaviate

from weaviate_client import client

CATEGORIES = [
    "Action",
    "Documentary",
    "Family",
    "Drama",
    "Horror",
    "Fantasy",
    "Adventure",
    "History",
    "Romance",
    "Music",
    "Western",
    "Animation",
    "War",
    "Comedy",
    "Mystery",
    "TV Movie",
    "Thriller",
    "Science Fiction",
    "Crime",
]

METADATA_FIELD_INFO = [
    AttributeInfo(
        name="genres",
        description="The genres of the movie. Must be one of the following: {', '.join(CATEGORIES)}}",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="release_year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="imdb_vote_average",
        description="A 1-10 rating for the movie",
        type="float",
    ),
]


def get_best_docs(input: str) -> list[Document]:
    document_content_description = "Brief summary of a movie"
    llm = OpenAI(temperature=0)

    vectorstore = Weaviate(
        client, "Movie", "text", attributes=["title", "show_id", "genres", "vote_count"]
    )
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        METADATA_FIELD_INFO,
        verbose=True,
    )
    return retriever.get_relevant_documents(input)
