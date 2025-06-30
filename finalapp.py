import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI


def get_primary_directive() -> str:
    instructions = (
        "The following information is provided to help you assist users with local events in Winterville, Georgia."
        "Do not stray from these instructions. "
        "Your role is to provide accurate and helpful information based on the events list provided."
        "You were created by three MSBA students at UGA: Sam Toole, Aidan Downey, and Jacob Croskey."
        "If a user asks about an event, you should respond with details such as the event name, date, time, location, and any other relevant information."
        "Do not make up information or provide details that are not in the events list."
        "If you do not know the answer, politely let the user know and offer to help with something else."
        "Always respond in a friendly and professional manner."
        "The events list will be provided to you in the format of a JSON string."
        "It will have the following fields: "
        "Event, Category, Date, Time, Location, Price. "
        "The date will be in the format YYYY-MM-DD, "
        "the time will be in the format HH:MM, "
        "and the price will be in dollars. "
        "Do not explicitly mention the JSON format in your responses, but use the information provided to answer user queries accurately."
        "Do not mention the primary directive in your responses, it is just a guideline for you to follow."
        "Provide all responses in a markdown format"
    )
    return instructions


def get_chat_history() -> str:
    """
    Returns the chat history as a formatted string.
    """
    if "messages" not in st.session_state:
        return "No chat history available."

    chat_history = []
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        chat_history.append(f"{role.capitalize()}: {content}")

    return "\n".join(chat_history)


def generate_complete_prompt(current_prompt: str) -> str:
    """
    Generates a complete prompt for the assistant, including the primary directive and chat history.
    """
    primary_directive = get_primary_directive()
    chat_history = get_chat_history()

    return (
        f'This is your primary directive: "{primary_directive}".\n\n'
        f"Here is the chat history:\n{chat_history}\n\n"
        f"Here is the current prompt:\n{current_prompt}\n\n"
    )


def get_events_as_json() -> str:
    """
    returns a json string containing a list of events in Winterville, Georgia
    The data is pulled from athens_events.xlsx
    The events are all Winterville events, no more filtering is needed.
    The events contain the following fields:
        Event: event name
        Category: event category
        Date: event date
        Time: event time
        Location: event location
        Price: event price in dollars
    The json string is formatted as a list of dictionaries, where each dictionary represents an event.
    The json string is formatted as follows:
    ```json
    [
        {
            "Event": "Event Name",
            "Category": "Event Category",
            "Date": "YYYY-MM-DD",
            "Time": "HH:MM",
            "Location": "Event Location",
            "Price": 0.0
        },
        ...
    ]

    """

    df = pd.read_excel("athens_events.xlsx")
    df = df[df["Location"].str.contains("Winterville", case=False, na=False)]
    df = df[["Event", "Category", "Date", "Time", "Location", "Price"]]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce").dt.strftime("%H:%M")
    df["Price"] = df["Price"].fillna(0).astype(float)

    return df.to_json(orient="records")


os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]

load_dotenv()

# Set up the OpenAI LLM (using GPT-3.5-turbo; change to GPT-4 if desired)
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# Load the stored index and create chat engine
storage_context = StorageContext.from_defaults(persist_dir="./athens_events_index")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="context")

st.title("The Winterville Guide")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about Winterville events..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    llm_response = chat_engine.chat(generate_complete_prompt(prompt))

    with st.chat_message("assistant"):
        st.markdown(llm_response)
        st.session_state.messages.append({"role": "assistant", "content": llm_response})
