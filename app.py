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

# Set the API key explicitly from Streamlit secrets (make sure your secrets.toml includes your key)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]

# Load environment variables from .env (optional, for local development) 
load_dotenv()

# Set up the OpenAI LLM (using GPT-3.5-turbo; change to GPT-4 if desired)
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# Load the stored index and create chat engine
storage_context = StorageContext.from_defaults(persist_dir="./athens_events_index")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="context")

# Load and parse events data
events_df = pd.read_excel("athens_events.xlsx")
events_df["Date"] = pd.to_datetime(events_df["Date"], errors="coerce").dt.date

st.title("The Winterville Guide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_target_date" not in st.session_state:
    st.session_state.last_target_date = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
    for _, row in df.iterrows():
        date_str = row["Date"].strftime("%A, %B %d, %Y") if row["Date"] else ""
        time_str = format_time_str(row["Time"])
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

def group_events_by_day(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No events found._"
    df = df.sort_values(["Date", "Time"])
    grouped_text = ""
    current_day = None
    for _, row in df.iterrows():
        day_date = row["Date"]
        if day_date != current_day:
            if grouped_text:
                grouped_text += "\n\n"
            day_str = day_date.strftime("%A, %B %d, %Y")
            grouped_text += f"**{day_str}**\n"
            current_day = day_date
        time_str = format_time_str(row["Time"])
        price_val = row.get("Price", 0)
        if pd.notnull(price_val):
            try:
                pval = float(price_val)
                price_str = "Free" if pval == 0 else f"${pval:.2f}"
            except:
                price_str = str(price_val)
        else:
            price_str = "Free"
        bullet_line = f"- {row['Event']} at {time_str} @ {row['Location']} ({price_str})"
        grouped_text += bullet_line + "\n"
    return grouped_text.strip()

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
    days_until_monday = (7 - today.weekday()) % 7 or 7
    next_monday = today + timedelta(days=days_until_monday)
    next_sunday = next_monday + timedelta(days=6)
    return next_monday, next_sunday

def get_this_week_range():
    today = datetime.now(tz).date()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week, end_of_week

def get_next_weekend():
    today = datetime.now(tz).date()
    days_until_saturday = (5 - today.weekday()) % 7
    saturday = today + timedelta(days=days_until_saturday)
    sunday = saturday + timedelta(days=1)
    return saturday, sunday

if prompt := st.chat_input("Ask me about Winterville events..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    prompt_lower = prompt.lower()
    category = None
    if "music" in prompt_lower:
        category = "Music"
    elif "comedy" in prompt_lower:
        category = "Comedy"
    elif "karaoke" in prompt_lower:
        category = "Karaoke & Open Mic"

    if "who made you" in prompt_lower or "who created you" in prompt_lower:
        direct_response = "I was created by three MSBA students at UGA: Sam Toole, Aidan Downey, and Jacob Croskey."
        with st.chat_message("assistant"):
            st.markdown(direct_response)
            st.session_state.messages.append({"role": "assistant", "content": direct_response})
        st.stop()

    elif "what is today" in prompt_lower:
        today_str = datetime.now(tz).strftime("%A, %B %d, %Y")
        response_text = f"Today is {today_str}!"
        with st.chat_message("assistant"):
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.stop()

    elif "this week" in prompt_lower:
        start_date, end_date = get_this_week_range()
        df = filter_events(category=category, start_date=start_date, end_date=end_date)
        events_text = group_events_by_day(df)
        dataset_context = f"Events for this week (Monday {start_date} → Sunday {end_date}):\n\n{events_text}"

    elif "next week" in prompt_lower:
        start_date, end_date = get_next_week_range()
        df = filter_events(category=category, start_date=start_date, end_date=end_date)
        events_text = group_events_by_day(df)
        dataset_context = f"Events for next week (Monday {start_date} → Sunday {end_date}):\n\n{events_text}"

    elif "weekend" in prompt_lower:
        start_date, end_date = get_next_weekend()
        df = filter_events(category=category, start_date=start_date, end_date=end_date)
        events_text = group_events_by_day(df)
        dataset_context = f"This weekend (Saturday {start_date} & Sunday {end_date}):\n\n{events_text}"

    else:
        location_substring = None
        location_match = re.search(r"events.*?(?:at|in)\s+([A-Za-z0-9&\-']+.*)", prompt_lower)
        if location_match:
            location_substring = location_match.group(1).strip()
        df = filter_events(category=category, location_substring=location_substring)
        events_text = format_events_simple_list(df)
        dataset_context = f"Here are upcoming events that match your query:\n\n{events_text}"

    with st.chat_message("assistant"):
        st.markdown(dataset_context)
        st.session_state.messages.append({"role": "assistant", "content": dataset_context})
