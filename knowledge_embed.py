import json
import os

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Instance untuk embedder
embedder = SentenceTransformer("BAAI/bge-m3")

db = mysql.connector.connect(
    host=os.getenv("TIDB_HOST"),
    port=int(os.getenv("TIDB_PORT")),
    user=os.getenv("TIDB_USER"),
    password=os.getenv("TIDB_PASSWORD"),
    database=os.getenv("TIDB_DATABASE"),
    ssl_ca=os.getenv("TIDB_SSL_CA"),
    ssl_verify_cert=True,
    ssl_verify_identity=True,
)

curr = db.cursor()

# Baca data knowledge

df = pd.read_csv("data_knowledge.csv")

for index, row in df.iterrows():
    text = str(row["Context_RAG"])

    try:
        embedding_list = embedder.encode(text).tolist()
        embedding_str = json.dumps(embedding_list)

        sql_query = """
            INSERT INTO documents (text, embedding) VALUES (%s, %s)
        """

        curr.execute(sql_query, (text, embedding_str))
        print(f"Successfully inserted row {index} into the database.")

    except (
        mysql.connector.Error,
        ValueError,
        TypeError,
        RuntimeError,
        UnicodeEncodeError,
    ) as e:
        print(f"Error processing row {index}: {e}")

db.commit()
curr.close()
print("All rows have been processed and inserted into the database.")
