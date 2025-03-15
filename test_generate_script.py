def generate_podcast_script(query_str: str) -> str:
    """
    Generates a podcast script based on a query.
    1. Retrieves relevant context from the vector DB.
    2. Constructs a detailed prompt for the LLM.
    3. Calls the OpenAI API to generate a full podcast script.
    """
    # Retrieve relevant documents
    collection = get_chroma_collection()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = embed_model.encode([query_str])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    docs = results["documents"][0] if results["documents"] else []
    if not docs:
        return "No relevant documents found."
    
    # Combine the retrieved docs into context. You may need to trim or summarize if too large.
    context_text = "\n\n".join(docs)
    
    # Build a detailed prompt for generating a podcast script.
    prompt = (
        "You are an experienced MLB podcast host. Your job is to generate a full podcast script "
        "based on the following context. The script should have a compelling introduction, a discussion "
        "of key points (including relevant statistics, team performance, and notable players), and a concise conclusion. "
        "Make the script engaging and conversational.\n\n"
        "Context:\n"
        f"{context_text}\n\n"
        "Based on this context, generate a complete podcast script."
    )
    
    # Prepare the message payload for ChatCompletion
    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced MLB podcast host who produces engaging and informative podcast episodes. "
                "Follow the instructions in the user's message to generate a detailed script."
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
        model="gpt-3.5-turbo",  # or 'gpt-4' if available
        messages=messages,
        temperature=0.7,
        max_tokens=1500  # Increase this if you want a longer script
    )
    
    podcast_script = response.choices[0].message.content.strip()
    return podcast_script

if __name__ == "__main__":
    query = "Generate a podcast script about MLB power rankings before opening day."
    script = generate_podcast_script(query)
    print("Generated Podcast Script:\n")
    print(script)
