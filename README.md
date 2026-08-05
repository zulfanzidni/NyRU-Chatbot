# ☕ NYRU Cafe Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed for **NYRU COFFEE & JUICE**, built using Streamlit, the Gemini API, and a TiDB Vector Database.

![NYRU Cafe Chatbot Screenshot](https://github.com/zulfanzidni/NyRU-Chatbot/blob/main/assets/ss.jpg)

## Features
- **Interactive Chat Interface**
- **RAG Architecture**
- **Powered by Gemini**
- **TiDB Vector Search**

## Tech Stack
- **Frontend / UI**: Streamlit
- **LLM**: Google Gemini (`google-generativeai`)
- **Embeddings**: Sentence Transformers (`BAAI/bge-m3`)
- **Database**: TiDB (MySQL compatible with Vector search capabilities)
- **Data Ingestion**: Pandas

## Local Setup

### Prerequisites
1. Python 3.12
2. A Gemini API Key from Google AI Studio.
3. A TiDB Cloud Serverless Cluster configured with a `documents` table for storing vector embeddings.

### Installation

1. **Clone the repository** (if applicable) and navigate to the project directory.

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows**: `.\.venv\Scripts\activate`
   - **Mac/Linux**: `source .venv/bin/activate`

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables**:
   Create a `.env` file in the root directory based on your credentials:
   ```env
   TIDB_HOST=your_tidb_host
   TIDB_PORT=4000
   TIDB_USER=your_tidb_user
   TIDB_PASSWORD=your_tidb_password
   TIDB_DATABASE=RAG
   TIDB_SSL_CA=isrgrootx1.pem
   GEMINI_API_KEY=your_gemini_api_key
   ```

### Running the App
Start the Streamlit application by running:
```bash
streamlit run app.py
```

## Deployment
This application is designed to be easily deployed to **Streamlit Community Cloud**. 
Simply connect your GitHub repository and configure your `.env` keys in the Streamlit Cloud **Secrets Management** dashboard.
