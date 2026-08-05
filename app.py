import json
import os

import google.generativeai as genai
import mysql.connector
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Page config
st.set_page_config(page_title="NYRU Cafe Chatbot", page_icon="☕")

# Cache the database connection and embedder to prevent reloading on every interaction
@st.cache_resource
def get_embedder():
    return SentenceTransformer("BAAI/bge-m3")


@st.cache_resource(ttl=300)
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("TIDB_HOST"),
        port=int(os.getenv("TIDB_PORT")),
        user=os.getenv("TIDB_USER"),
        password=os.getenv("TIDB_PASSWORD"),
        database=os.getenv("TIDB_DATABASE"),
        ssl_ca=os.getenv("TIDB_SSL_CA"),
        ssl_verify_cert=True,
        ssl_verify_identity=True,
    )


def search_document(database, embedder, query, k_top=10):
    results = []
    query_embedding_list = embedder.encode(query).tolist()
    query_embedding_str = json.dumps(query_embedding_list)

    # Ping the database to ensure connection is alive, reconnect if necessary
    try:
        database.ping(reconnect=True, attempts=3, delay=1)
    except Exception as e:
        st.error(f"Failed to reconnect to database: {e}")
        return []

    curr = database.cursor()
    sql_query = f"""
        SELECT text, vec_cosine_distance(embedding, %s) AS distance
        FROM documents
        ORDER BY distance ASC
        LIMIT {k_top};
    """
    
    try:
        curr.execute(sql_query, (query_embedding_str,))
        search_results = curr.fetchall()
        for result in search_results:
            text, distance = result
            results.append({"text": text, "distance": distance})
    except Exception as e:
        st.error(f"Database search error: {e}")
    finally:
        database.commit()
        curr.close()
        
    return results


def response_query(database, embedder, query):
    retrieved_doc = search_document(database, embedder, query)
    context = "\n".join([doc["text"] for doc in retrieved_doc])

    prompt = (
        f"Answer the question based on the context below. If the answer is not contained "
        f"within the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"


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
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = response_query(db, embedder, prompt)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
