import asyncio
import json
import os
from enum import Enum
from typing import Any

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from tmdbv3api import Discover, Movie, TMDb


class Providers(str, Enum):
    AppleTV = "2"
    Netflix = "8"
    AmazonPrimeVideo = "9"
    AmazonVideo = "10"
    DisenyPlus = "337"


def _prepare_parameters(year: int, page: int, providers: list[str]) -> dict[str, Any]:
    return {
        "sort_by": "popularity.desc",
        "watch_region": "US",
        "with_watch_providers": "|".join(providers),
        "page": page,
        "year": year,
    }


def fetch(
    year_range: tuple[int, int], page_range: tuple[int, int], providers: list[str]
) -> list[dict]:
    results_collection = []
    for year in range(year_range[0], year_range[1] + 1):
        for page in range(page_range[0], page_range[1] + 1):
            parameters = _prepare_parameters(year, page, providers)
            new_results = discover.discover_movies(parameters)["results"]
            results_collection.extend(new_results)
    return results_collection


def _get_ids(res: list[dict[str, Any]]) -> list[int]:
    return [r.get("id") for r in res]


def _build_headers(api_key: str) -> dict[str, str]:
    return {"accept": "application/json", "Authorization": f"Bearer {api_key}"}


async def fetch_movie_details(
    session, movie_id: int, headers: dict[str, str], semaphore: asyncio.Semaphore
):
    async with semaphore:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?append_to_response=videos%2Cwatch%2Fproviders&language=en-US"
        async with session.get(url, headers=headers) as response:
            return await response.text()


async def main(movie_ids, headers: dict[str, str], max_concurrent_requests=20):
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_movie_details(session, movie_id, headers, semaphore)
            for movie_id in movie_ids
        ]
        responses = await asyncio.gather(*tasks)
        return responses


def to_pandas(results: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.json_normalize(results, max_level=1)


if __name__ == "__main__":
    PROVIDERS = [Providers.Netflix.value, Providers.DisenyPlus.value]
    DATE_RANGE = (2020, 2023)
    PAGES = (1, 1)
    load_dotenv()
    
    tmdb = TMDb()
    tmdb.api_key = os.environ["TMDB_API_KEY"]
    movie = Movie()
    discover = Discover()

    search_results = fetch(DATE_RANGE, PAGES, PROVIDERS)
    movie_ids = _get_ids(search_results)

    access_token = os.environ["TMDB_ACCESS_TOKEN"]
    if access_token:
        headers = _build_headers(access_token)
        details = asyncio.run(main(movie_ids, headers, max_concurrent_requests=10))
        parsed_details = [json.loads(result) for result in details]
        movies = to_pandas(parsed_details)
        movies.to_parquet("data/final_movies.parquet")
