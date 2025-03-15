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

# 1) Load environment variables
from dotenv import load_dotenv
load_dotenv()  # This looks for a .env file in the current working directory

# 2) Retrieve values or set defaults
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GOOGLE_CRED_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

def scrape_article(url: str) -> dict:
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
    """
    Uploads a list of article dictionaries as JSON to the specified GCS bucket.
    """
    # The client will pick up credentials from the env variable GOOGLE_APPLICATION_CREDENTIALS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    blob.upload_from_string(
        data=json.dumps(data, indent=2),
        content_type="application/json"
    )
    print(f"Uploaded {blob_name} to gs://{bucket_name}.")

def get_chroma_collection(persist_dir=CHROMA_PERSIST_DIR):

    # Using the new style client
    client = chromadb.PersistentClient(path=persist_dir)

    collection_name = "mlb_articles"
    # If the collection doesn’t exist, create it;
    # if it does, get it.
    collection = client.get_or_create_collection(collection_name)

    return collection


def embed_and_insert(articles: list, collection, model_name=EMBEDDING_MODEL_NAME):
    """
    Embeds the article text using a local model and inserts into Chroma.
    """
    model = SentenceTransformer(model_name)
    texts = [art["body"] for art in articles]
    embeddings = model.encode(texts)

    ids = [art["url"] for art in articles]  # or any unique ID logic
    metadatas = articles  # can store any additional fields

    collection.delete(ids=["https://www.mlb.com/news/jurrangelo-cijntje-pitches-lefty-and-righty-in-spring-breakout?t=mlb-pipeline-coverage"])

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    print("Articles embedded and inserted into Chroma DB.")

def test_query(collection, query_str: str, model_name=EMBEDDING_MODEL_NAME):
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

if __name__ == "__main__":
    # Example article URLs
    article_urls = [
        #"https://www.mlb.com/news/jurrangelo-cijntje-pitches-lefty-and-righty-in-spring-breakout?t=mlb-pipeline-coverage",
        "https://www.mlb.com/news/mlb-power-rankings-before-opening-day-2025"
    ]
    
    scraped_data = []
    for url in article_urls:
        print(f"Scraping: {url}")
        try:
            article = scrape_article(url)
            scraped_data.append(article)
            time.sleep(2)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    
    # Example filename: articles/2025-03-15/batch.json
    blob_name = f"articles/{datetime.now().strftime('%Y-%m-%d')}/articles_batch.json"
    store_in_gcs(scraped_data, BUCKET_NAME, blob_name)

    collection = get_chroma_collection()
    embed_and_insert(scraped_data, collection)

    #test_query(collection, "Who is the 2nd team in MLB's power rankings?")
