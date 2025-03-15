import streamlit as st
import os
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI

# Set the API key explicitly from Streamlit secrets (make sure your secrets.toml includes your key)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]

# Load environment variables from .env (optional, for local development) 
load_dotenv()

# Set up the OpenAI LLM (using GPT-3.5-turbo; change to GPT-4 if desired)
Settings.llm = OpenAI(model="gpt-4")

# Load the stored index built from your Athens events dataset
storage_context = StorageContext.from_defaults(persist_dir="./athens_events_index")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="context")

# Load the events dataset from Excel and parse the Date column
events_df = pd.read_excel("athens_events.xlsx")
events_df["Date"] = pd.to_datetime(events_df["Date"])

# Set the app title to "The Guide Dawg 🐾"
st.title("The Guide Dawg 🐾")

# Initialize conversation history and last_target_date in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_target_date" not in st.session_state:
    st.session_state.last_target_date = None

# Display previous conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Helper Functions ---

def format_price(price):
    try:
        price_value = float(price)
        return "Free" if price_value == 0 else f"${price_value:.2f}"
    except Exception:
        return str(price)

def format_time(time_str):
    try:
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        return time_obj.strftime("%-I:%M %p")
    except Exception:
        return time_str

def get_events_for_date_range(target_date):
    filtered = events_df[events_df["Date"].dt.date == target_date.date()]
    if filtered.empty:
        return "No events found for this date."
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        event_date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str_formatted = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']} — {price_str}\n"
    return context

def get_events_for_category_and_date(category, target_date):
    filtered = events_df[(events_df["Category"].str.lower() == category.lower()) &
                         (events_df["Date"].dt.date == target_date.date())]
    if filtered.empty:
        return f"No {category} events found on {target_date.strftime('%A, %B %d, %Y')}."
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        event_date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str_formatted = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']} — {price_str}\n"
    return context

def get_category_events_for_date_range_range(category, start_date, end_date):
    filtered = events_df[(events_df["Category"].str.lower() == category.lower()) &
                         (events_df["Date"].dt.date >= start_date.date()) &
                         (events_df["Date"].dt.date <= end_date.date())]
    if filtered.empty:
        return f"No {category} events found from {start_date.strftime('%A, %B %d, %Y')} to {end_date.strftime('%A, %B %d, %Y')}."
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        event_date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str_formatted = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']} — {price_str}\n"
    return context

def get_events_for_date_range_range(start_date, end_date):
    filtered = events_df[(events_df["Date"].dt.date >= start_date.date()) &
                         (events_df["Date"].dt.date <= end_date.date())]
    if filtered.empty:
        return "No events found for this period."
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        event_date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str_formatted = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']} — {price_str}\n"
    return context

# New helper functions to group events by day over a date range.
def get_grouped_events_for_date_range_range(start_date, end_date):
    current = start_date
    all_context = ""
    while current <= end_date:
        events = get_events_for_date_range(current)
        day_str = current.strftime("%A, %B %d, %Y")
        if events.startswith("No events"):
            events_text = "No events."
        else:
            events_text = events
        all_context += f"{day_str}:\n{events_text}\n"
        current += timedelta(days=1)
    return all_context

def get_grouped_category_events_for_date_range(category, start_date, end_date):
    current = start_date
    all_context = ""
    while current <= end_date:
        events = get_events_for_category_and_date(category, current)
        day_str = current.strftime("%A, %B %d, %Y")
        if events.startswith(f"No {category}"):
            events_text = "No events."
        else:
            events_text = events
        all_context += f"{day_str}:\n{events_text}\n"
        current += timedelta(days=1)
    return all_context

def determine_target_date(query, base_date):
    query_lower = query.lower()
    # Explicit check for "tomorrow"
    if "tomorrow" in query_lower:
        return base_date + timedelta(days=1)
    # Reuse last target date if query uses terms like "that day", "later", etc.
    if st.session_state.get("last_target_date") and any(term in query_lower for term in ["that day", "later", "other", "that night"]):
        return st.session_state["last_target_date"]
    if "next week" in query_lower:
        next_monday = base_date + timedelta(days=(7 - base_date.weekday()))
        return next_monday
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for day in days:
        if day in query_lower:
            today_index = base_date.weekday()
            target_index = days.index(day)
            days_ahead = target_index - today_index if target_index >= today_index else target_index - today_index + 7
            return base_date + timedelta(days=days_ahead)
    match = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", query)
    if match:
        date_str = match.group(1)
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
    if "weekend" in query_lower:
        return base_date + timedelta(days=(5 - base_date.weekday()))
    return base_date

def build_dataset_context(query, target_date):
    query_lower = query.lower()
    if "next week" in query_lower:
        next_sunday = target_date + timedelta(days=6)
        if "karaoke" in query_lower:
            return get_grouped_category_events_for_date_range("Karaoke & Open Mic", target_date, next_sunday)
        elif "music" in query_lower or "concert" in query_lower:
            return get_grouped_category_events_for_date_range("Music", target_date, next_sunday)
        elif "comedy" in query_lower:
            return get_grouped_category_events_for_date_range("Comedy", target_date, next_sunday)
        else:
            return get_grouped_events_for_date_range_range(target_date, next_sunday)
    if "karaoke" in query_lower:
        return get_events_for_category_and_date("Karaoke & Open Mic", target_date)
    elif "music" in query_lower or "concert" in query_lower:
        return get_events_for_category_and_date("Music", target_date)
    elif "comedy" in query_lower:
        return get_events_for_category_and_date("Comedy", target_date)
    else:
        return get_events_for_date_range(target_date)

# --- Main Interaction ---

current_date = datetime.today()
today_str = current_date.strftime("%A, %B %d, %Y")

weekday = current_date.weekday()
this_saturday = current_date + timedelta(days=(5 - weekday))
this_sunday = current_date + timedelta(days=(6 - weekday))
weekend_str = f"{this_saturday.strftime('%A, %B %d, %Y')} to {this_sunday.strftime('%A, %B %d, %Y')}"

if prompt := st.chat_input("Ask me about Athens events or plan a date:"):
    target_date = determine_target_date(prompt, current_date)
    st.session_state["last_target_date"] = target_date
    dataset_context = build_dataset_context(prompt, target_date)
    
    query_lower = prompt.lower()
    if "next week" in query_lower:
        next_monday = current_date + timedelta(days=(7 - current_date.weekday()))
        next_sunday = next_monday + timedelta(days=6)
        date_context_text = f"for next week (Monday: {next_monday.strftime('%A, %B %d, %Y')} to Sunday: {next_sunday.strftime('%A, %B %d, %Y')})"
    elif any(term in query_lower for term in ["saturday", "sunday", "that day", "later", "other"]):
        date_context_text = f"for {target_date.strftime('%A, %B %d, %Y')}"
    elif "weekend" in query_lower:
        date_context_text = f"for the weekend (Saturday: {this_saturday.strftime('%A, %B %d, %Y')}, Sunday: {this_sunday.strftime('%A, %B %d, %Y')})"
    else:
        date_context_text = f"for {target_date.strftime('%A, %B %d, %Y')}"
    
    conversation_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])

    custom_instructions = (
        f"Hey, it's {today_str} and we're in the Eastern Time Zone. {date_context_text}. "
        "You're The Guide Dawg 🐾—your chill, collegiate event and date planning assistant with access to the Athens events dataset. "
        "When someone asks 'What are you?', say: 'I am The Guide Dawg, your go-to resource for local Athens events.' "
        "If someone asks 'Who made you?' or 'Who created you?', say: 'I was created by three MSBA students at UGA: Sam Toole, Aidan Downey, and Jacob Croskey.' "
        "When someone asks 'What is your purpose?', say: 'My purpose is to help UGA students and the broader Athens community easily discover local events, enriching the campus experience and fostering a vibrant, connected community.' "
        "For queries that are purely informational—such as 'What's going on on this day?' or 'What all events are happening tomorrow?'—simply list all the events for that day without additional recommendations, organized by event type if possible. "
        "If a query refers to 'this weekend', list events separately for Saturday and Sunday, indicating if one day has no events. "
        "If a query mentions a specific location (e.g., 'What events are happening at The Foundry?'), list all events corresponding to that location from the dataset. "
        "For 'next week' queries, provide a list of events grouped by day in chronological order. "
        "For other queries, base your responses primarily on the dataset provided below and arrange events in strict chronological order "
        "(morning events first, then afternoon, then evening; use standard times like '8:00 AM'). "
        "Differentiate between providing a curated itinerary for date planning and simply listing events when the user wants to know what's happening. "
        "If asked to plan a a date night specifically, combine an event or two later in the day with a dinner recoomendation, combine your creativity with the dataset. "
        "However, if the query is casual or conversational (e.g., 'what's up', 'how's it going', 'whats going on'), respond naturally with a friendly greeting and creative flair. "
        "Ensure that the final output is nicely organized, consistently formatted, and uses the same plain text font throughout. Your goal is to improve the student and local experience in Athens GA, be helpful!"
        "Below is the dataset context for the specified period:\n"
        f"{dataset_context}"
    )
    
    full_prompt = (
        f"{custom_instructions}\n\n"
        f"Conversation History:\n{conversation_history}\n\n"
        f"User: {prompt}\n"
        f"Assistant (in a chill tone):"
    )
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = chat_engine.chat(full_prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

