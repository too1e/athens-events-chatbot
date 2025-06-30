import streamlit as st
import os
from datetime import datetime
import re
from dotenv import load_dotenv
import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
import pytz

# Fix PermissionError by setting a custom tiktoken cache directory
os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"

# Ensure timezone is set
tz = pytz.timezone('America/New_York')
today_str = datetime.now(tz).strftime("%A, %B %d, %Y")

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

if prompt := st.chat_input("Ask me about Winterville events..."):
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

    # ✅ NEW: filter market category if applicable
    filtered_df = events_df.copy()
    if "market" in prompt_lower or "markets" in prompt_lower:
        filtered_df = filtered_df[filtered_df["category"].str.lower() == "markets"]

    events_text = format_events_simple_list(filtered_df)

    final_query = f"""
You're The Winterville Guide — a helpful local chatbot for events in Winterville.

Today is {today_str} Eastern Time. When the user asks about dates like "next weekend", "this Friday", or "two weeks from now", always interpret those dates based on the current date — not the dataset. Use reasoning to figure out what exact dates they mean, even if the user doesn't specify a number. If you're unsure, it's okay to ask the user to clarify. Never assume the wrong date range.

If the user asks for the "next market", always return the soonest upcoming event from any event categorized as "Markets", including both "Marigold Farmers Market" and "Marigold Monday Market", and make sure it is the earliest by date but the date has not passed.

If the user asks about events for next week, list all events from the next week, which is between the next sunday and saturday from today. 

Here is a list of all upcoming events:

{events_text}

When the user asks a question, try your best to interpret the date, topic, or location, and suggest matching events if they exist. Do not hallucinate or make up events, only give out information if you can verify it from the events text.

Remember the ongoing chat context. You have access to the full conversation history above. Use prior questions or topics the user has asked in this session to give smarter, more personalized responses.

User asked: {prompt}
"""

    llm_response = chat_engine.chat(final_query)

    with st.chat_message("assistant"):
        st.markdown(llm_response)
        st.session_state.messages.append({"role": "assistant", "content": llm_response})
