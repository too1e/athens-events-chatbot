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
import requests
from ics import Calendar

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

def load_and_clean_ical_events():
    url = "https://calendar.google.com/calendar/ical/wintervillecityhall%40gmail.com/public/basic.ics"
    response = requests.get(url)
    cal = Calendar(response.text)

    rows = []
    for event in cal.events:
        rows.append({
            "event": event.name,
            "category": "Government",  # Can be improved with keyword classification
            "date": event.begin.date(),
            "time": event.begin.time(),
            "location": event.location.split(",")[0] if event.location else "TBD",
            "price": 0
        })

    df = pd.DataFrame(rows)
    return df

# Load and merge local and iCal events
df_local = pd.read_excel("athens_events.xlsx")
df_local.columns = df_local.columns.str.strip().str.lower()
df_local["date"] = pd.to_datetime(df_local["date"], errors="coerce").dt.date

df_ical = load_and_clean_ical_events()
df_ical.columns = df_ical.columns.str.strip().str.lower()

events_df = pd.concat([df_local, df_ical], ignore_index=True).drop_duplicates(subset=["event", "date", "time"])

st.title("The Winterville Guide")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
    df = df.sort_values(["date", "time"])
    lines = []
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%A, %B %d, %Y") if row["date"] else ""
        time_str = format_time_str(row["time"])
        price_val = row.get("price", 0)
        if pd.notnull(price_val):
            try:
                pval = float(price_val)
                price_str = "Free" if pval == 0 else f"${pval:.2f}"
            except:
                price_str = str(price_val)
        else:
            price_str = "Free"
        line = f"- {row['event']} on {date_str} at {time_str} @ {row['location']} ({price_str})"
        lines.append(line)
    return "\n".join(lines)

def group_events_by_day(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No events found._"
    df = df.sort_values(["date", "time"])
    grouped_text = ""
    current_day = None
    for _, row in df.iterrows():
        day_date = row["date"]
        if day_date != current_day:
            if grouped_text:
                grouped_text += "\n\n"
            day_str = day_date.strftime("%A, %B %d, %Y")
            grouped_text += f"**{day_str}**\n"
            current_day = day_date
        time_str = format_time_str(row["time"])
        price_val = row.get("price", 0)
        if pd.notnull(price_val):
            try:
                pval = float(price_val)
                price_str = "Free" if pval == 0 else f"${pval:.2f}"
            except:
                price_str = str(price_val)
        else:
            price_str = "Free"
        bullet_line = f"- {row['event']} at {time_str} @ {row['location']} ({price_str})"
        grouped_text += bullet_line + "\n"
    return grouped_text.strip()

def filter_events(category=None, start_date=None, end_date=None, location_substring=None) -> pd.DataFrame:
    df = events_df.copy()
    if category:
        df = df[df["category"].fillna("").str.lower() == category.lower()]
    if location_substring:
        df = df[df["location"].fillna("").str.lower().str.contains(location_substring.lower())]
    if start_date and end_date:
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    elif start_date:
        df = df[df["date"] == start_date]
    return df

def get_this_week_range():
    today = datetime.now(tz).date()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week, end_of_week

def get_next_week_range():
    today = datetime.now(tz).date()
    next_monday = today + timedelta(days=(7 - today.weekday()))
    next_sunday = next_monday + timedelta(days=6)
    return next_monday, next_sunday

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

    start_date = end_date = None
    if "what is today" in prompt_lower:
        today_str = datetime.now(tz).strftime("%A, %B %d, %Y")
        dataset_context = f"Today is {today_str}."

    elif "this week" in prompt_lower:
        start_date, end_date = get_this_week_range()
        df = filter_events(category=category, start_date=start_date, end_date=end_date)
        dataset_context = group_events_by_day(df)

    elif "next week" in prompt_lower:
        start_date, end_date = get_next_week_range()
        df = filter_events(category=category, start_date=start_date, end_date=end_date)
        dataset_context = group_events_by_day(df)

    elif "weekend" in prompt_lower:
        start_date, end_date = get_next_weekend()
        df = filter_events(category=category, start_date=start_date, end_date=end_date)
        dataset_context = group_events_by_day(df)

    else:
        location_substring = None
        location_match = re.search(r"events.*?(?:at|in)\s+([A-Za-z0-9&\-']+.*)", prompt_lower)
        if location_match:
            location_substring = location_match.group(1).strip()
        df = filter_events(category=category, location_substring=location_substring)
        dataset_context = format_events_simple_list(df)

    final_query = f"You are The Winterville Guide — an assistant for local events.\n\nUser asked: {prompt}\n\nRelevant events:\n{dataset_context}"
    llm_response = chat_engine.chat(final_query)

    with st.chat_message("assistant"):
        st.markdown(llm_response)
        st.session_state.messages.append({"role": "assistant", "content": llm_response})
