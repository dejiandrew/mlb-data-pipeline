import os
import openai
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

##########################
# LOAD ENV / CONFIG VARS #
##########################
load_dotenv("/workspaces/mlb-data-pipeline/.env")  # Reads your .env if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY. Please set it in .env or environment variables.")

openai.api_key = OPENAI_API_KEY

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

##########################
# INITIALIZE CHROMA      #
##########################
def get_chroma_collection(path=CHROMA_PERSIST_DIR, collection_name="mlb_articles"):
    # Using the new style Chroma client
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(collection_name)
    return collection

##########################
# RETRIEVAL FUNCTION     #
##########################
def retrieve_docs(query: str, collection, n_results=3):
    """
    Embed the query locally, retrieve top docs from Chroma.
    Returns a list of document strings.
    """
    # 1) Load local embedding model
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # 2) Encode the query
    query_embedding = embed_model.encode([query])[0]
    
    # 3) Query Chroma for top docs
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    
    # results["documents"] is a list of lists; we just want the first list
    docs = results["documents"][0] if results["documents"] else []
    
    return docs

##########################
# OPENAI CHAT COMPLETION #
##########################
def generate_answer_with_openai(user_query: str, context_snippets: list) -> str:
    """
    Takes the user query + some context snippets, calls OpenAI ChatCompletion.
    Returns the LLM's final answer as a string.
    """
    # Combine context into one string
    context_text = "\n\n".join(context_snippets)
    
    # Build the messages array for ChatCompletion
    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable MLB baseball analyst. "
                "Use the provided context to answer the user's question. "
                "If the context is insufficient, say 'I do not know.'"
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"
        }
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",    # or 'gpt-4' if you have access
        messages=messages,
        temperature=0.7,
        max_tokens=600,          # how long you want the answer to be
    )
    
    # Extract the text from the response
    return response["choices"][0]["message"]["content"].strip()

##########################
# MAIN RAG PIPELINE      #
##########################
def rag_pipeline(user_query: str):
    # 1. Get Chroma collection
    collection = get_chroma_collection()
    
    # 2. Retrieve relevant docs
    docs = retrieve_docs(user_query, collection)
    if not docs:
        return "No relevant documents found in Chroma."

    # 3. Call OpenAI to generate final answer
    final_answer = generate_answer_with_openai(user_query, docs)
    return final_answer

##########################
# DEMO USAGE             #
##########################
if __name__ == "__main__":
    user_query = "Who is the 2nd team in MLB's power rankings?"
    answer = rag_pipeline(user_query)
    print("\nFinal Answer:\n", answer)