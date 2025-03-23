import os
import json
import time
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from google.cloud import storage
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv("/workspaces/mlb-data-pipeline/.env")  # Loads variables from .env

# Retrieve values or set defaults
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GOOGLE_CRED_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

import chromadb
from chromadb.config import Settings

def scrape_article(url: str) -> dict:
    """Scrape an article from a given URL."""
    resp = requests.get(url)
    resp.encoding = 'utf-8'
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
    print("🗑️ Clearing Chroma DB...")
    collection.delete(where={"id": {"$ne": ""}})
    print("✅ Cleared. Current count:", len(collection.get()["ids"]))
    model = SentenceTransformer(model_name)
    texts = [art["body"] for art in articles]
    embeddings = model.encode(texts)
    ids = [art["url"] for art in articles]  # Unique ID logic based on URL
    metadatas = articles
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
      3. Instantiates an OpenAI client and calls the API to generate a final answer.
    """
    # Retrieve relevant documents using local embeddings
    collection = get_chroma_collection()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = model.encode([query_str])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    docs = results["documents"][0] if results["documents"] else []
    if not docs:
        return "No relevant documents found."
    context_text = "\n\n".join(docs)
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
    
    # Instantiate the OpenAI client inside the function
    from openai import OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    
    response = client.chat.completions.create(
        model="gpt-4",  # or 'gpt-4' if available
        messages=messages,
        temperature=0.7,
        max_tokens=600
    )
    final_answer = response.choices[0].message.content.strip()
    return final_answer

import logging

def generate_podcast_script(query_str: str) -> str:
    """
    Generates a podcast script based on a query.
    1. Retrieves relevant context from the vector DB.
    2. Constructs a detailed prompt for the LLM.
    3. Calls the OpenAI API to generate a full podcast script.
    """
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Retrieve relevant documents
    collection = get_chroma_collection()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = embed_model.encode([query_str])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3, include=["documents", "metadatas"])

    docs = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    if not docs:
        logger.warning("No relevant documents found.")
        return "No relevant documents found."

    # Log the retrieved documents and metadata
    logger.info(f"Query to Chroma: {query_str}")
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        logger.info(f"\n--- Top Doc #{i + 1} ---")
        logger.info(f"Title: {meta.get('title', 'N/A')}")
        logger.info(f"URL: {meta.get('url', 'N/A')}")
        logger.info(f"Excerpt: {doc[:500]}...\n")

    # Combine the retrieved docs into context
    context_text = "\n\n".join(docs)
    logger.info("=== Final Context Passed to LLM ===")
    logger.info(context_text[:2000])

    # Build a detailed prompt for generating a podcast script
    prompt = (
        "You are an experienced MLB podcast host. Generate a natural, conversational podcast script "
        "about the following baseball information.\n\n"
        "Important guidelines:\n"
        "1. Write this as a fluid monologue without any 'Host:' prefixes\n"
        "2. Don't include any section headings like [INTRO] or [CONCLUSION]\n"
        "3. Create natural transitions between topics instead of labeling sections\n"
        "4. Make it sound like you're having a casual conversation with the listener\n"
        "5. Include your enthusiasm for baseball but keep it professional\n"
        "6. Begin with a brief greeting and introduction to the topic\n"
        "7. End with a sign-off that teases future content\n\n"
        "Context:\n"
        f"{context_text}\n\n"
        "Based on this context, generate a complete, natural-sounding podcast script."
    )

    # Prepare the message payload for ChatCompletion
    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced MLB podcast host who produces engaging and informative podcast episodes. "
                "Create natural, flowing monologues without any speaker labels or section headings."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Instantiate OpenAI client inside the function
    from openai import OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)

    response = openai_client.chat.completions.create(
        #model="gpt-3.5-turbo",
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=1500
    )

    podcast_script = response.choices[0].message.content.strip()
    return podcast_script

# def format_script_for_tts(script):
#     """Format a script for better text-to-speech results"""
#     # Remove any remaining section headers or speaker indicators
#     formatted = re.sub(r'\[(.*?)\]', '', script)
#     formatted = re.sub(r'Host:', '', formatted)
    
#     # Replace common baseball abbreviations and terms
#     replacements = {
#         "MLB": "M L B",
#         "HR": "home run",
#         "RBI": "R B I",
#         "ERA": "E R A",
#         "AL": "A L",
#         "NL": "N L",
#         "vs.": "versus",
#         "vs": "versus",
#         "Phillies": "Fillies",  # Help with pronunciation
#     }
    
#     for term, replacement in replacements.items():
#         formatted = formatted.replace(f" {term} ", f" {replacement} ")
#         # Also catch terms at the beginning of sentences
#         formatted = formatted.replace(f"{term} ", f"{replacement} ")
    
#     # Match ordinal numbers (1st, 2nd, 3rd, etc.)
#     formatted = re.sub(r'(\d+)(st|nd|rd|th)', r'\1 \2', formatted)
    
#     # Add subtle pauses after sentences using actual pauses instead of [break]
#     formatted = formatted.replace(". ", ". ... ")
#     formatted = formatted.replace("! ", "! ... ")
#     formatted = formatted.replace("? ", "? ... ")
    
#     # Handle paragraph breaks with pauses
#     formatted = formatted.replace("\n\n", "\n...\n")
    
#     return formatted

def format_script_for_tts(script: str) -> str:
    """
    Clean and optimize the podcast script for text-to-speech.
    Removes markdown, excess whitespace, or formatting symbols like '****'
    """
    # Remove lines that are just asterisks or other formatting
    cleaned_lines = []
    for line in script.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"[*\-_=]{3,}", stripped):
            continue  # skip lines with only *** or --- or ___
        cleaned_lines.append(stripped)

    return " ".join(cleaned_lines)


import os
import requests

def generate_audio_with_your_voice(script_text, output_file="podcast_output.mp3"):
    """Generate audio using your cloned voice on ElevenLabs"""
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    api_key = os.getenv("ELEVENLABS_API_KEY")
    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")  # Use env var or fallback
    
    if not voice_id or not api_key:
        raise ValueError("Missing ELEVENLABS_VOICE_ID or ELEVENLABS_API_KEY in environment variables")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    data = {
        "text": script_text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 0.8,
            "style": 0.25,
            "use_speaker_boost": True
        }
    }
    
    print(f"Generating audio for script ({len(script_text)} characters)...")
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"✅ Audio generated: {output_file}")
        return output_file
    else:
        print(f"❌ Error generating audio: {response.status_code}")
        print(response.text)
        return None

def upload_podcast_to_gcs(file_path, bucket_name, blob_name=None):
    """Upload podcast audio or script file to GCS"""
    if not blob_name:
        # If no blob name provided, use the filename with a logical path
        file_type = "audio" if file_path.endswith(".mp3") else "scripts"
        filename = os.path.basename(file_path)
        blob_name = f"podcasts/{file_type}/{filename}"
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Detect content type based on file extension
    content_type = "audio/mpeg" if file_path.endswith(".mp3") else "text/plain"
    
    # Upload file
    blob.upload_from_filename(file_path, content_type=content_type)
    print(f"Uploaded {file_path} to gs://{bucket_name}/{blob_name}")
    
    return f"gs://{bucket_name}/{blob_name}"