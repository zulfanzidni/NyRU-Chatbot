import json
import os

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Instance untuk embedder ringan
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
print(f"Loading embedding model: {MODEL_NAME}...")
embedder = SentenceTransformer(MODEL_NAME)

db = mysql.connector.connect(
    host=os.getenv("TIDB_HOST"),
    port=int(os.getenv("TIDB_PORT") or 4000),
    user=os.getenv("TIDB_USER"),
    password=os.getenv("TIDB_PASSWORD"),
    database=os.getenv("TIDB_DATABASE"),
    ssl_ca=os.getenv("TIDB_SSL_CA"),
    ssl_verify_cert=True,
    ssl_verify_identity=True,
)

curr = db.cursor()

# Siapkan tabel documents dengan vector(384)
print("Recreating documents table with VECTOR(384)...")
curr.execute("DROP TABLE IF EXISTS documents;")
curr.execute("""
    CREATE TABLE documents (
        id INT AUTO_INCREMENT PRIMARY KEY,
        text TEXT,
        embedding VECTOR(384)
    );
""")

# Baca data knowledge
df = pd.read_csv("data_knowledge.csv")
texts = [str(row["Context_RAG"]) for _, row in df.iterrows()]

print(f"Encoding {len(texts)} texts in batch...")
embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)

data = []
for text, emb in zip(texts, embeddings):
    data.append((text, json.dumps(emb.tolist())))

sql_query = "INSERT INTO documents (text, embedding) VALUES (%s, %s)"
curr.executemany(sql_query, data)

db.commit()
curr.close()
db.close()
print(f"Successfully processed and inserted {len(data)} rows into TiDB.")
