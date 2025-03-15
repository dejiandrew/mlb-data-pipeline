import os
import json
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from google.cloud import storage
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings

from dotenv import load_dotenv
load_dotenv("/workspaces/mlb-data-pipeline/.env")  # Loads variables from .env

# Retrieve values or set defaults
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GOOGLE_CRED_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

def scrape_article(url: str) -> dict:
    """Scrape an article from a given URL."""
    resp = requests.get(url)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, "html.parser")
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"
    body = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
    
    return {
        "url": url,
        "title": title,
        "body": body,
        "scraped_at": datetime.utcnow().isoformat()
    }

def store_in_gcs(data: list, bucket_name: str, blob_name: str):
    """Upload a list of article dictionaries as JSON to the specified GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    blob.upload_from_string(
        data=json.dumps(data, indent=2),
        content_type="application/json"
    )
    print(f"Uploaded {blob_name} to gs://{bucket_name}.")

def get_chroma_collection(persist_dir=CHROMA_PERSIST_DIR):
    """Initialize a persistent Chroma client and return the collection."""
    client = chromadb.PersistentClient(path=persist_dir)
    collection_name = "mlb_articles"
    collection = client.get_or_create_collection(collection_name)
    return collection

def embed_and_insert(articles: list, collection, model_name=EMBEDDING_MODEL_NAME):
    """
    Embed the article texts using a local SentenceTransformer model
    and insert them into the Chroma collection.
    """
    model = SentenceTransformer(model_name)
    texts = [art["body"] for art in articles]
    embeddings = model.encode(texts)
    ids = [art["url"] for art in articles]  # Unique ID logic based on URL
    metadatas = articles

    # Optionally delete a specific ID if needed:
    # collection.delete(ids=["https://example.com/old-article"])

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    print("Articles embedded and inserted into Chroma DB.")

def test_query(collection, query_str: str, model_name=EMBEDDING_MODEL_NAME):
    """
    Query the Chroma collection with a provided question, and print the top results.
    """
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query_str])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    print(f"\nQuery: {query_str}")
    print("Top Results:")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print("-----")
        print("Title:", meta["title"])
        print("URL:", meta["url"])
        print("Excerpt:", doc[:150], "...")
    print("-----")

def rag_pipeline(query_str: str) -> str:
    """
    Runs a full RAG pipeline:
      1. Retrieves relevant documents from Chroma.
      2. Constructs a prompt with those documents.
      3. Calls the OpenAI API to generate a final answer.
    """
    # Get the Chroma collection and retrieve relevant docs.
    collection = get_chroma_collection()
    # We'll retrieve a few documents (you can tweak n_results as needed).
    # For simplicity, assume test_query is replaced by our retrieval function.
    # We'll use test_query's logic to retrieve the docs.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = model.encode([query_str])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    docs = results["documents"][0] if results["documents"] else []
    
    if not docs:
        return "No relevant documents found."

    # Build context string from docs. You might want to limit the length if needed.
    context_text = "\n\n".join(docs)
    
    # Construct the messages for the OpenAI ChatCompletion
    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable MLB baseball analyst. "
                "Use the provided context to answer the user's question. "
                "If the context does not contain enough information, say 'I do not know.'"
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query_str}\nAnswer:"
        }
    ]
    
    # Call OpenAI API
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    openai.api_key = openai_api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or 'gpt-4' if available
        messages=messages,
        temperature=0.7,
        max_tokens=600
    )
    
    final_answer = response["choices"][0]["message"]["content"].strip()
    return final_answer
