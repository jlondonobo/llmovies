import json
import os
from typing import Literal

import openai
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


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
Your response will help filter some results, so don't say anything!

If the user asks you anything different than movies or TV shows, respectfully stop the conversation.
"""


def generate_response(input: str) -> TopicInput:
    response = openai.ChatCompletion.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": input},
        ],
    )
    return response["choices"][0]["message"]["content"]


def parse_response(response: str) -> TopicInput:
    try:
        as_json = json.loads(response)
        topic_input = TopicInput(**as_json)
        return topic_input
    except json.decoder.JSONDecodeError:
        print(response)
        raise LLMoviesOutputError("The response from the chatbot was not valid JSON.")
    except ValidationError as e:
        print(e.errors())
        raise LLMoviesOutputError

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "What would you like to watch?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        
if user_input := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(user_input) 
            try:
                parsed = parse_response(response)
            except LLMoviesOutputError:
                st.write(response)
            
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
    
try:
    st.write(parsed)
except NameError:
    pass
    