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

# Fix PermissionError for tiktoken cache
os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"

# Timezone setup
tz = pytz.timezone('America/New_York')

# Load API key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]

load_dotenv()
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# Load LlamaIndex
storage_context = StorageContext.from_defaults(persist_dir="./athens_events_index")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="context")

# Load events data
events_df = pd.read_excel("athens_events.xlsx")
events_df.columns = events_df.columns.str.strip().str.lower()
events_df["date"] = pd.to_datetime(events_df["date"], errors="coerce").dt.date

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
        if isinstance(time_val, datetime):
            return time_val.strftime("%-I:%M %p")
        if isinstance(time_val, str) and ("AM" in time_val.upper() or "PM" in time_val.upper()):
            return pd.to_datetime(time_val).strftime("%-I:%M %p")
        return datetime.strptime(str(time_val), "%H:%M:%S").strftime("%-I:%M %p")
    except:
        return str(time_val)

def group_events_by_day(df):
    if df.empty:
        return "_No events found._"
    df = df.sort_values(["date", "time"])
    output = ""
    current_day = None
    for _, row in df.iterrows():
        if row["date"] != current_day:
            if output:
                output += "\n\n"
            output += f"**{row['date'].strftime('%A, %B %d, %Y')}**\n"
            current_day = row["date"]
        time_str = format_time_str(row["time"])
        price_val = row.get("price", 0)
        price_str = "Free" if pd.isna(price_val) or price_val == 0 else f"${float(price_val):.2f}"
        output += f"- {row['event']} at {time_str} @ {row['location']} ({price_str})\n"
    return output.strip()

def filter_events(start_date=None, end_date=None, category=None):
    df = events_df.copy()
    if start_date and end_date:
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    elif start_date:
        df = df[df["date"] == start_date]
    if category:
        df = df[df["category"].fillna("").str.lower() == category.lower()]
    return df

def get_week_range(weeks_ahead=0):
    today = datetime.now(tz).date()
    monday = today - timedelta(days=today.weekday()) + timedelta(weeks=weeks_ahead)
    sunday = monday + timedelta(days=6)
    return monday, sunday

def interpret_prompt(prompt):
    prompt = prompt.lower()
    today = datetime.now(tz).date()
    if "today" in prompt:
        return filter_events(start_date=today)
    elif "tomorrow" in prompt:
        return filter_events(start_date=today + timedelta(days=1))
    elif "this week" in prompt or ("going on" in prompt and "week" in prompt):
        start, end = get_week_range()
        return filter_events(start_date=start, end_date=end)
    elif "next week" in prompt:
        start, end = get_week_range(1)
        return filter_events(start_date=start, end_date=end)
    elif "weekend" in prompt:
        saturday = today + timedelta((5 - today.weekday()) % 7)
        sunday = saturday + timedelta(days=1)
        return filter_events(start_date=saturday, end_date=sunday)
    elif match := re.search(r"july \d+|\d+/\d+/\d+", prompt):
        try:
            date_obj = pd.to_datetime(match.group(0), errors="coerce").date()
            return filter_events(start_date=date_obj)
        except:
            pass
    return events_df[events_df["date"] >= today]  # Default: future events

if prompt := st.chat_input("Ask me about Winterville events..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    df = interpret_prompt(prompt)
    events_text = group_events_by_day(df)

    final_query = f"""
You are The Winterville Guide — an AI assistant that ONLY answers using the list of Winterville events shown below.

User asked: \"{prompt}\"

If there are matching events in the list, summarize them.
If none match, say politely that there are no events that fit.

Event list:
{events_text}
"""

    llm_response = chat_engine.chat(final_query)

    with st.chat_message("assistant"):
        st.markdown(llm_response)
        st.session_state.messages.append({"role": "assistant", "content": llm_response})
