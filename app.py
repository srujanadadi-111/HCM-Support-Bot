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
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

# Streamlit interface
import streamlit as st

# Initialize session state for tracking question clicks
if 'question_clicks' not in st.session_state:
    st.session_state.question_clicks = {
        "What is HCM?": 0,
        "How do you do custom report generation?": 0,
        "What are the features of HCM?": 0,
        "What are the steps involved in creating a KPI?": 0,
        "What is a data view?": 0
    }

def handle_trending_click(question):
    # Update click count in session state
    st.session_state.question_clicks[question] += 1
    # Set the clicked question as the current query
    st.session_state.query = question
    return question

# Main interface
st.image("egovlogo.png", width=200)
st.title("HCM Support Bot [Beta version]")

# Trending Questions Section
st.subheader("Trending Questions")
col1, col2 = st.columns(2)

# First column of trending questions
with col1:
    for question in list(st.session_state.question_clicks.keys())[:3]:
        if st.button(f"📈 {question}", key=f"btn_{question}"):
            query = handle_trending_click(question)

# Second column of trending questions
with col2:
    for question in list(st.session_state.question_clicks.keys())[3:]:
        if st.button(f"📈 {question}", key=f"btn_{question}"):
            query = handle_trending_click(question)

# User input section
query = st.text_input("Ask a question:", key="query")
submit_button = st.button("Submit")

if submit_button:
    if query.strip():
        st.write("Query Received:", query)
        answer = chat_with_assistant(query)
        st.write(f"Assistant's answer: {answer}")
    else:
        st.warning("Please enter a question before clicking Submit.")

# Optional: Display trending metrics
if st.checkbox("Show Trending Metrics"):
    st.write("Question Click Counts:")
    sorted_questions = dict(sorted(
        st.session_state.question_clicks.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    for q, count in sorted_questions.items():
        st.write(f"- {q}: {count} clicks")
