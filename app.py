import os

import mysql.connector
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from config import get_llm_config
from rag import (
    build_chat_messages,
    get_openai_client,
    search_document,
    stream_chat_response,
)

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="NYRU Cafe Chatbot", page_icon="☕")


# Cache the database connection, embedder, and LLM client to prevent reloading on every interaction
@st.cache_resource
def get_embedder():
    return SentenceTransformer("BAAI/bge-m3")


@st.cache_resource(ttl=300)
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("TIDB_HOST"),
        port=int(os.getenv("TIDB_PORT") or 4000),
        user=os.getenv("TIDB_USER"),
        password=os.getenv("TIDB_PASSWORD"),
        database=os.getenv("TIDB_DATABASE"),
        ssl_ca=os.getenv("TIDB_SSL_CA"),
        ssl_verify_cert=True,
        ssl_verify_identity=True,
    )


@st.cache_resource
def get_client():
    return get_openai_client()


@st.cache_resource
def get_config():
    return get_llm_config()


# --- Streamlit UI ---

st.title("☕ NYRU COFFEE & JUICE Assistant")
st.write("Welcome to NYRU Cafe! Ask me anything about our menu, promos, or location.")

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load resources
try:
    embedder = get_embedder()
    db = get_db_connection()
    client = get_client()
    config = get_config()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Retrieve relevant docs from TiDB and build messages
    search_results = search_document(db, embedder, prompt)
    context_docs = [doc["text"] for doc in search_results]
    messages = build_chat_messages(context_docs, prompt, history=st.session_state.messages)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response via streaming
    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                stream_chat_response(client, config["model"], messages)
            )
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Gagal menghubungi layanan AI: {e}")
            if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
                st.session_state.messages.pop()

