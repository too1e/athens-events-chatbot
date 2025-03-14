import streamlit as st

from datetime import datetime, timedelta

from dotenv import load_dotenv

import pandas as pd

from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Document

from llama_index.core.settings import Settings

from llama_index.llms.openai import OpenAI

import dateparser

import os



# Load API key from .env file

load_dotenv()



# Set up the OpenAI LLM (using GPT-4)

Settings.llm = OpenAI(model="gpt-4")



# Load the events dataset

events_df = pd.read_excel("athens_events.xlsx")

events_df["Date"] = pd.to_datetime(events_df["Date"])



def load_or_rebuild_index():

    index_dir = "./athens_events_index"

    if os.path.exists(index_dir) and os.listdir(index_dir):

        # Load existing index

        storage_context = StorageContext.from_defaults(persist_dir=index_dir)

        index = load_index_from_storage(storage_context)

    else:

        # Rebuild index from DataFrame

        documents = []

        for _, row in events_df.iterrows():

            text = f"""Event: {row['Event']}

Time: {row['Time']}

Location: {row['Location']}

Price: {row['Price']}"""

            metadata = {

                "Category": row["Category"],

                "Date": row["Date"].strftime("%Y-%m-%d"),

                "Price": row["Price"],

                "Location": row["Location"],

            }

            documents.append(Document(text=text, metadata=metadata))

        

        index = VectorStoreIndex.from_documents(documents)

        index.storage_context.persist(persist_dir=index_dir)

    return index



index = load_or_rebuild_index()

chat_engine = index.as_chat_engine(chat_mode="context")



st.title("The Athens Passport 🗺️")



# Initialize conversation history

if "messages" not in st.session_state:

    st.session_state.messages = []



# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])



# Date handling functions

def get_current_date():

    return datetime.now()



def get_weekend_dates(date):

    weekday = date.weekday()

    saturday = date + timedelta(days=(5 - weekday))

    sunday = date + timedelta(days=(6 - weekday))

    return saturday, sunday



def parse_date_from_input(user_input):

    try:

        parsed_date = dateparser.parse(user_input, settings={"PREFER_DATES_FROM": "future"})

        if not parsed_date:

            st.warning("Could not parse the date from your input. Please try again with a specific date or time frame.")

            return None

        return parsed_date

    except Exception as e:

        st.warning(f"Error parsing date: {e}")

        return None



# Event display functions

def format_event_for_display(row):

    return (

        f"• **{row['Event']}** on {row['Date'].strftime('%A, %B %d, %Y')} at {row['Time']} "

        f"at {row['Location']}. Price: {row['Price']}"

    )



def filter_events_by_date(df, start_date, end_date):

    mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)

    filtered = df.loc[mask]

    return [format_event_for_display(row) for _, row in filtered.iterrows()]



# Custom instructions

current_date = get_current_date()

today_str = current_date.strftime("%A, %B %d, %Y")

date_plan_instructions = (

    f"You are The Athens Passport, a highly creative and helpful assistant for UGA students looking for events and date ideas in Athens. Today is {today_str}. "

    "Your PRIMARY and MOST IMPORTANT goal is to create creative and engaging DATE PLANS. "

    "When a user asks to 'plan a date', you MUST create a detailed and imaginative date itinerary. "

    "Do NOT simply list events. You must combine events from the dataset with your own unique ideas. "

    "Always consider the user's preferences and tailor your suggestions. "

    "If no events are available, use your creativity to suggest alternative activities or locations. "

    "Provide responses in a friendly, collegiate tone."

)

event_query_instructions = (

    f"You are The Athens Passport, an event information assistant for UGA students. Today is {today_str}. "

    "Use the provided events dataset to answer questions about events. "

    "Provide accurate and up-to-date information about events. "

    "If no events are available, inform the user. "

    "Provide responses in a friendly, collegiate tone."

)



# User input and response

if prompt := st.chat_input("What's on your mind about Athens events?"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):

        st.markdown(prompt)



    # Get the current date and weekend dates

    current_date = get_current_date()

    saturday, sunday = get_weekend_dates(current_date)



    if "plan a date" in prompt.lower():

        with st.chat_message("assistant"):

            st.markdown("Let me create a date plan for you!")

            try:

                # Filter events for the weekend

                weekend_events = filter_events_by_date(events_df, saturday, sunday)



                if not weekend_events:

                    st.markdown("No events found for this weekend. Here are some alternative suggestions:")

                    response = chat_engine.chat(date_plan_instructions + " User request: " + prompt)

                    st.markdown(response.response)

                    st.session_state.messages.append({"role": "assistant", "content": response.response})

                else:

                    # Combine events into a creative date idea

                    event_list = "\n".join(weekend_events)

                    creative_prompt = (

                        f"Here are some events happening this weekend:\n{event_list}\n"

                        "Combine these events into a creative and engaging date plan. "

                        "Provide a detailed itinerary with a mix of events and your own unique ideas. "

                        "Make it fun and memorable!"

                    )

                    response = chat_engine.chat(creative_prompt)

                    st.markdown(response.response)

                    st.session_state.messages.append({"role": "assistant", "content": response.response})

                

                st.session_state.date_plan_context = True

            except Exception as e:

                st.markdown(f"Sorry, I encountered an error: {e}")

    elif st.session_state.get("date_plan_context"):

        with st.chat_message("assistant"):

            st.markdown("Updating your date plan!")

            try:

                response = chat_engine.chat(date_plan_instructions + " User request: " + prompt + ". Update the current date plan.")

                st.markdown(response.response)

                st.session_state.messages.append({"role": "assistant", "content": response.response})

            except Exception as e:

                st.markdown(f"Sorry, I encountered an error: {e}")

    else:

        with st.chat_message("assistant"):

            try:

                response = chat_engine.chat(event_query_instructions + " User request: " + prompt)

                st.markdown(response.response)

                st.session_state.messages.append({"role": "assistant", "content": response.response})

                st.session_state.date_plan_context = False

            except Exception as e:

                st.markdown(f"Sorry, I encountered an error: {e}")



    if "plan a date" not in prompt.lower():

        st.session_state.date_plan_context = False
