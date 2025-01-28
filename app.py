import pickle
import os
import numpy as np
import openai
import nltk
import streamlit as st

# Load the document store from the file
with open('document_store.pkl', 'rb') as f:
    document_store = pickle.load(f)

# Cosine similarity function for comparing embeddings
def cosine_similarity(vec1, vec2):
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    if vec1_norm == 0 or vec2_norm == 0:
        return 0.0

    return np.dot(vec1, vec2) / (vec1_norm * vec2_norm)

# Retrieve relevant chunks for a query based on cosine similarity
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = generate_embeddings([query])[0]
    similarities = []

    for doc_name, doc_data in document_store.items():
        for chunk, chunk_embedding in zip(doc_data["chunks"], doc_data["embeddings"]):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk, similarity, doc_name))

    relevant_chunks = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunk, doc_name) for chunk, _, doc_name in relevant_chunks]

# Process and upload documents from the downloaded folder
def process_and_upload_from_folder(folder_path, doc_type, chunking_strategy):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf") and doc_type == "pdf":
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".pptx") and doc_type == "ppt":
            text = extract_text_from_ppt(file_path)
        elif filename.endswith(".xlsx") and doc_type == "excel":
            text = extract_text_from_excel(file_path)
        else:
            continue

        # Clean and chunk the text
        clean_doc_text = clean_text(text)
        text_chunks = chunking_strategy(clean_doc_text)
        embeddings = generate_embeddings(text_chunks)

        document_store[filename] = {
            "chunks": text_chunks[:len(embeddings)],
            "embeddings": embeddings,
            "source": doc_type
        }
        print(f"Uploaded {filename} ({doc_type.upper()}) to the memory store.")

# Chat with the assistant based on the query
def chat_with_assistant(query):
    relevant_chunks = retrieve_relevant_chunks(query)
    context = "\n\n".join([f"Source ({doc}): {chunk}" for chunk, doc in relevant_chunks])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": """You are a precise assistant that answers questions based strictly on the provided context.
            Rules:
            1. Use ONLY information from the context
            2. Keep exact terminology and steps from the source
            3. If multiple sources have different information, specify which source you're using
            4. If information isn't in the context, say "I don't have enough information"
            5. For procedures, list exact steps in order
            6. Include specific buttons, links, and UI elements mentioned in the source"""
        }, {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }],
        temperature=0,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

# Process documents from the folder and then interact with the assistant
def process_and_upload_all(folder_path):
    process_and_upload_from_folder(folder_path, "pdf", split_text_into_chunks)
    process_and_upload_from_folder(folder_path, "ppt", split_text_into_chunks)
    process_and_upload_from_folder(folder_path, "excel", split_text_into_chunks)

# Streamlit interface
process_and_upload_all(folder_path)
st.write("Documents processed and uploaded.")

st.title("Document Processing and Chat Assistant")

# User input for querying
query = st.text_input("Ask a question:")
if query:
    answer = chat_with_assistant(query)
    st.write(f"Assistant's answer: {answer}")

