import pickle
import os
import numpy as np
import openai
import streamlit as st


# Load the document store from the file
try:
    with open('document_store (3).pkl', 'rb') as f:
        document_store = pickle.load(f)
    #st.write("Document store loaded successfully!")
except FileNotFoundError:
    st.write("Error: The document store file 'document_store.pkl' was not found.")
    document_store = {}

# Setup OpenAI API Key
openai.api_key = st.secrets["openai_api_key"]

# Cosine similarity function for comparing embeddings
def cosine_similarity(vec1, vec2):
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    if vec1_norm == 0 or vec2_norm == 0:
        return 0.0

    return np.dot(vec1, vec2) / (vec1_norm * vec2_norm)

def generate_embeddings(texts, batch_size=10):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=batch
            )
            embeddings.extend([embedding["embedding"] for embedding in response["data"]])
        except Exception as e:
            print(f"Error generating embeddings for batch {i}-{i+batch_size}: {e}")
    return embeddings

def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = generate_embeddings([query])[0]
    similarities = []

    for doc_name, doc_data in document_store.items():
        for chunk, chunk_embedding in zip(doc_data["chunks"], doc_data["embeddings"]):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk, similarity, doc_name))

    relevant_chunks = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunk, doc_name) for chunk, _, doc_name in relevant_chunks]

def chat_with_assistant(query):
    relevant_chunks = retrieve_relevant_chunks(query)
    print(f"Relevant chunks for query '{query}':")
    for chunk, doc in relevant_chunks:
        print(f"Source ({doc}): {chunk}")
    context = "\n\n".join([f"Source ({doc}): {chunk}" for chunk, doc in relevant_chunks])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """You are a precise assistant that answers questions based strictly on the provided context.
                Rules:
                1. Use ONLY information from the context
                2. Keep exact terminology and steps from the source
                3. If multiple sources have different information, specify which source you're using
                4. If information isn't in the context, say "I don't have enough information"
                5. For procedures, list exact steps in order
                6. Include specific buttons, links, and UI elements mentioned in the source"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        temperature=0,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

# Streamlit interface
st.title("HCM Support Bot")

# User input for querying
query = st.text_input("Ask a question:")
if query:
    st.write("Query Received:", query)  # Show the query entered
    answer = chat_with_assistant(query)
    st.write(f"Assistant's answer: {answer}")
