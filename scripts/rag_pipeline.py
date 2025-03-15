import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

####################
# CONFIG VARIABLES #
####################
from dotenv import load_dotenv
load_dotenv("/workspaces/mlb-data-pipeline/.env")   # Or an absolute path to your .env file

CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR")  # Path to your local Chroma DB
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Choose a small open-source model. "facebook/blenderbot-400M-distill" is purely for demonstration;
# It's not specialized in baseball knowledge, but it’s light enough to run on CPU.
LLM_MODEL_NAME = "facebook/blenderbot-400M-distill"


##############
# SETUP LLM  #
##############

# We create a pipeline for text generation using a local model
# (This is a Seq2Seq type model, so we use 'text2text-generation' or 'conversational' pipelines.)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


##################
# RAG PIPELINE   #
##################

def get_chroma_collection(chroma_path=CHROMA_PATH):
    # Use the new Chroma client style
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection("mlb_articles")
    return collection

def retrieve_context(user_query: str, collection, n_results=3):
    # Convert query to embedding
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vector = embed_model.encode([user_query])[0]

    # Retrieve from Chroma
    results = collection.query(query_embeddings=[query_vector], n_results=n_results)
    # We only care about the top docs
    documents = results["documents"][0]  # list of strings
    # Could also retrieve metadata with: results["metadatas"][0]
    return documents

def generate_answer_with_context(user_query: str, context_snippets: list) -> str:
    combined_context = "\n\n".join(context_snippets)
    prompt = (
        f"Context:\n{combined_context}\n\n"
        f"Question: {user_query}\n\n"
        "Answer in a helpful style: "
    )
    print("FINAL PROMPT:", prompt)

    # 1) Tokenize prompt with explicit truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100)

    # 2) Double-check how many tokens we have
    input_length = inputs["input_ids"].shape[1]
    print("Tokenized prompt length:", input_length)  # e.g., 100

    # 3) Generate with a small "max_new_tokens" 
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,  # generate up to 64 new tokens
        do_sample=True,
    )

    # 4) Decode and return
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text


############
# MAIN     #
############

if __name__ == "__main__":
    # Example user query
    user_query = "Who is the 2nd team in MLB's power rankings?"
    
    # 1. Connect to local Chroma DB
    collection = get_chroma_collection()
    print("Total docs in collection:", collection.count())

    # 2. Retrieve top relevant context
    context_snippets = retrieve_context(user_query, collection)
    if not context_snippets:
        print("No relevant documents found.")
    else:
        print("Context Snippets:")
        for snippet in context_snippets:
            print("------")
            print(snippet[:200], "...")
        print("------\n")

    # 3. Generate LLM answer
    final_answer = generate_answer_with_context(user_query, context_snippets)
    print("Answer:\n", final_answer)
