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

#header
st.markdown("""
    <style>
        div[data-testid="stHeader"] {
            display: none;
        }
        .sticky-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 9999;
            background-color: #0e1117;
            padding: 0.5rem 1rem;
            border-bottom: 1px solid #333;
        }
        .sticky-header h1 {
            margin: 0;
            font-size: 1.75rem;
            color: white;
        }
        .main {
            padding-top: 4rem;
        }
    </style>
    <div class="sticky-header">
        <h1>The Winterville Guide</h1>
    </div>
""", unsafe_allow_html=True)


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

    events_text = format_events_simple_list(events_df)

    final_query = f"""
You're The Winterville Guide — a helpful local chatbot for events in Winterville.

Today is {today_str} Eastern Time.

If the user asks about "this week", "next week", or similar phrases, define them using the ISO weekday standard:

- "This week" = from Monday to Sunday of the current week (based on today's date)
- "Next week" = the Monday–Sunday block *after* the current week
- "This weekend" = the upcoming Saturday and Sunday
- "Next weekend" = the Saturday and Sunday of the following week

Always interpret phrases like "this Friday" or "two weeks from now" relative to today, using correct calendar math.

Use only verified events in the list below. Do not guess or hallucinate.

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
