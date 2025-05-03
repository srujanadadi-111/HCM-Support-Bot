import pickle
import os
import numpy as np
import openai
import streamlit as st
from pymongo import MongoClient

# Load the document store from the file
try:
    with open('documen_store.pkl', 'rb') as f:
        document_store = pickle.load(f)
    #st.write("Document store loaded successfully!")
except FileNotFoundError:
    st.write("Error: The document store file 'documen_store.pkl' was not found.")
    document_store = {}
except Exception as e:
    st.error(f"Error loading document store: {e}")
    document_store = {}
    
# Setup OpenAI API Key
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, AttributeError, RuntimeError, Exception):
    openai.api_key = os.environ.get("OPENAI_API_KEY")

# Cosine similarity function for comparing embeddings with improved numerical stability
def cosine_similarity(vec1, vec2):
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    if vec1_norm < 1e-10 or vec2_norm < 1e-10:
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

# Function to check if content is relevant
def is_relevant_content(chunk, query):
    """Check if chunk contains actual relevant information rather than metadata."""
    # Skip chunks that are mostly version numbers or deployment info
    if chunk.count(':v') > 3 or chunk.count('-') > 10:
        return False
    
    # Extract key terms from query (excluding common words)
    query_terms = set(term.lower() for term in query.split()
                     if term.lower() not in {'how', 'what', 'when', 'where', 'do', 'does', 'is', 'are', 'the'})

    # Check if chunk contains any query terms
    chunk_lower = chunk.lower()
    terms_found = sum(1 for term in query_terms if term in chunk_lower)

    return terms_found > 0

# Updated retrieve_relevant_chunks function with hybrid approach
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = generate_embeddings([query])[0]
    similarities = []
    
    # Extract key terms from query for keyword matching
    query_terms = set(term.lower() for term in query.split() 
                     if term.lower() not in {'how', 'what', 'when', 'where', 'do', 'does', 'is', 'are', 'the'})
    
    for doc_name, doc_data in document_store.items():
        for chunk, chunk_embedding in zip(doc_data["chunks"], doc_data["embeddings"]):
            # Calculate vector similarity
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            
            # Calculate keyword match score
            chunk_lower = chunk.lower()
            keyword_matches = sum(1 for term in query_terms if term in chunk_lower)
            
            # Boost score for slack PDFs if query seems to target them
            source_boost = 0.1 if doc_data.get("source") == "slack_pdf" and "slack" in query.lower() else 0
            
            # Combine scores (vector similarity is primary, keywords provide a boost)
            combined_score = similarity + (keyword_matches * 0.05) + source_boost
            
            similarities.append((chunk, combined_score, doc_name))

    # Get top results by combined score
    relevant_chunks = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunk, doc_name) for chunk, _, doc_name in relevant_chunks]

def chat_with_assistant(query):
    with st.spinner("Retrieving relevant information..."):
        relevant_chunks = retrieve_relevant_chunks(query)
        print(f"Relevant chunks for query '{query}':")
        for chunk, doc in relevant_chunks:
            print(f"Source ({doc}): {chunk}")
        context = "\n\n".join([f"Source ({doc}): {chunk}" for chunk, doc in relevant_chunks])

    with st.spinner("Generating response..."):
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
            max_tokens=1000
        )
    
    answer = response.choices[0].message.content.strip()

    # Store query and response in MongoDB
    #collection.insert_one({
    #   "query": query,
    #  "response": answer
    #})

    return answer

# Streamlit interface
# Initialize session state for tracking question clicks
if 'question_clicks' not in st.session_state:
    st.session_state.question_clicks = {
        "What is Health Campaign Management?": 0,
        "What are the steps involved in creating a KPI?": 0
    }

def handle_trending_click(question):
    # Update click count in session state
    st.session_state.question_clicks[question] += 1
    # Set the clicked question as the current query
    st.session_state.query = question
    return question

# Main interface
st.image("egovlogo.png", width=200)
st.title("Health Campaign Management (HCM) Support Bot [Beta version]")

# Notes Section
st.subheader("Note:")
st.markdown(
    '<p style="color:red; font-size:16px;">Please try to be as in detail as possible with your prompt and use full forms for beta version, e.g., Health Campaign Management instead of HCM.</p>',
    unsafe_allow_html=True,
)

# Trending Questions Section
st.subheader("Trending Questions")
col1, col2 = st.columns(2)

# First column of trending questions
with col1:
    for question in list(st.session_state.question_clicks.keys())[:1]:
        if st.button(f"📈 {question}", key=f"btn_{question}"):
            query = handle_trending_click(question)

# Second column of trending questions
with col2:
    for question in list(st.session_state.question_clicks.keys())[1:]:
        if st.button(f"📈 {question}", key=f"btn_{question}"):
            query = handle_trending_click(question)

# User input section
query = st.text_input("Ask a question:", key="query")
submit_button = st.button("Submit")

if submit_button:
    if query.strip():
        try:
            st.write("Query Received:", query)
            answer = chat_with_assistant(query)
            st.write(f"Assistant's answer: {answer}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question before clicking Submit.")
