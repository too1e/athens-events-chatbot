import streamlit as st
import os
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
import pytz

# 🛠 Fix PermissionError by setting a custom tiktoken cache directory
os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"

# Ensure timezone is set
tz = pytz.timezone('America/New_York')

# Set the API key explicitly from Streamlit secrets
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]

# Load environment variables from .env
load_dotenv()

# Set up the OpenAI LLM (using GPT-4)
Settings.llm = OpenAI(model="gpt-4")

# Lazy-load the index AFTER the first prompt to avoid hanging on app startup
index = None
chat_engine = None

# Load and parse events data
events_df = pd.read_excel("athens_events.xlsx")
events_df["Date"] = pd.to_datetime(events_df["Date"], errors="coerce").dt.date

st.title("The Guide Dawg 🐾")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_target_date" not in st.session_state:
    st.session_state.last_target_date = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Time and formatting helpers

def format_time_str(time_val):
    if pd.isnull(time_val):
        return ""
    try:
        return datetime.strptime(str(time_val), "%H:%M:%S").strftime("%-I:%M %p")
    except:
        return str(time_val)

def format_events_simple_list(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No events found._"
    df = df.sort_values(["Date","Time"])
    lines = []
    used_times = set()
    for _, row in df.iterrows():
        date_str = row["Date"].strftime("%A, %B %d, %Y") if row["Date"] else ""
        time_str = format_time_str(row["Time"])
        key = (date_str, time_str)
        if key in used_times:
            continue
        used_times.add(key)
        price_val = row.get("Price", 0)
        if pd.notnull(price_val):
            try:
                pval = float(price_val)
                price_str = "Free" if pval == 0 else f"${pval:.2f}"
            except:
                price_str = str(price_val)
        else:
            price_str = "Free"
        line = f"- {row['Event']} on {date_str} at {time_str} @ {row['Location']} ({price_str})"
        lines.append(line)
    return "\n".join(lines)

def filter_events(category=None, start_date=None, end_date=None, location_substring=None) -> pd.DataFrame:
    df = events_df.copy()
    if category:
        df = df[df["Category"].fillna("").str.lower() == category.lower()]
    if location_substring:
        df = df[df["Location"].fillna("").str.lower().str.contains(location_substring.lower())]
    if start_date and end_date:
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    elif start_date:
        df = df[df["Date"] == start_date]
    return df

def get_next_week_range():
    today = datetime.now(tz).date()
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    next_monday = today + timedelta(days=days_until_monday)
    return next_monday, next_monday + timedelta(days=6)

def get_next_weekend():
    today = datetime.now(tz).date()
    days_until_saturday = (5 - today.weekday()) % 7
    saturday = today + timedelta(days=days_until_saturday)
    return saturday, saturday + timedelta(days=1)

def parse_day_of_week(prompt_text: str):
    prompt_lower = prompt_text.lower()
    today = datetime.now(tz).date()
    if "tomorrow" in prompt_lower:
        return today + timedelta(days=1)
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(days):
        if day in prompt_lower:
            day_diff = (i - today.weekday()) % 7
            if day_diff == 0:
                day_diff = 7
            return today + timedelta(days=day_diff)
    return None

if prompt := st.chat_input("Ask me about Athens events or plan a date..."):
    # Lazy load index and chat engine on first use
    if chat_engine is None:
        storage_context = StorageContext.from_defaults(persist_dir="./athens_events_index")
        index = load_index_from_storage(storage_context)
        chat_engine = index.as_chat_engine(chat_mode="context")

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    prompt_lower = prompt.lower()
    if "who made you" in prompt_lower or "who created you" in prompt_lower:
        direct_response = "I was created by three MSBA students at UGA: Sam Toole, Aidan Downey, and Jacob Croskey."
        with st.chat_message("assistant"):
            st.markdown(direct_response)
            st.session_state.messages.append({"role": "assistant", "content": direct_response})
        st.stop()

    wants_date_plan = ("plan a date" in prompt_lower or "date night" in prompt_lower)

    if "what is today" in prompt_lower:
        today_str = datetime.now(tz).strftime("%A, %B %d, %Y")
        response_text = f"Today is {today_str}!"
        with st.chat_message("assistant"):
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.stop()

    elif "next week" in prompt_lower:
        category = None
        if "music" in prompt_lower:
            category = "Music"
        elif "comedy" in prompt_lower:
            category = "Comedy"
        elif "karaoke" in prompt_lower:
            category = "Karaoke & Open Mic"

        next_monday, next_sunday = get_next_week_range()
        df_next_week = filter_events(category=category, start_date=next_monday, end_date=next_sunday)
        dataset_context = f"Events for next week ({next_monday} - {next_sunday}):\n\n" + format_events_simple_list(df_next_week)

    elif "weekend" in prompt_lower:
        category = None
        if "music" in prompt_lower:
            category = "Music"
        elif "comedy" in prompt_lower:
            category = "Comedy"
        elif "karaoke" in prompt_lower:
            category = "Karaoke & Open Mic"

        sat, sun = get_next_weekend()
        df_weekend = filter_events(category=category, start_date=sat, end_date=sun)
        dataset_context = f"This weekend ({sat} & {sun}):\n\n" + format_events_simple_list(df_weekend)

    else:
        category = None
        if "music" in prompt_lower:
            category = "Music"
        elif "comedy" in prompt_lower:
            category = "Comedy"
        elif "karaoke" in prompt_lower:
            category = "Karaoke & Open Mic"

        location_substring = None
        location_match = re.search(r'events.*?(?:at|in)\s+([A-Za-z0-9\&\-\']+.*)', prompt_lower)
        if location_match:
            location_substring = location_match.group(1).strip()

        if wants_date_plan:
            day_date = parse_day_of_week(prompt)
            if not day_date:
                day_date = datetime.today().date() + timedelta(days=1)
            df_day = filter_events(category=category, start_date=day_date, end_date=day_date, location_substring=location_substring)
            trimmed_events_text = "\n".join(format_events_simple_list(df_day).splitlines()[:20])
            date_str = day_date.strftime("%A, %B %d, %Y")
            dataset_context = (
                f"You want a creative date plan for {date_str}.\n\n"
                f"Below is a list of events happening that day:\n\n{trimmed_events_text}"
            )
        else:
            df_upcoming = filter_events(category=category, location_substring=location_substring)
            dataset_context = "Here are the upcoming events:\n\n" + format_events_simple_list(df_upcoming)

    today_str = datetime.now(tz).strftime("%A, %B %d, %Y")
    extra_date_instructions = (
        "The user wants a creative date plan. Recommend a realistic itinerary using events that don't overlap in time. "
        "Space them out appropriately, and include one dinner suggestion from your own knowledge. Be fun, casual, and unique."
        if wants_date_plan else ""
    )

    custom_instructions = (
        f"Hey, it's {today_str} in the Eastern Time Zone. You're The Guide Dawg 🐾, a friendly event and date planning chatbot for UGA students. "
        f"{extra_date_instructions}\n\nBelow is the dataset context:\n{dataset_context}"
    )

    final_query = f"{custom_instructions}\n\nUser's prompt: {prompt}\n\nAssistant:"

    llm_response = chat_engine.chat(final_query)

    with st.chat_message("assistant"):
        st.markdown(llm_response)
        st.session_state.messages.append({"role": "assistant", "content": llm_response})
