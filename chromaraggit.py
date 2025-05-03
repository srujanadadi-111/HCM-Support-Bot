import openai
import chromadb
import fitz  # for pymupdf
import pptx  # for python-pptx
import openpyxl
import boto3
import os
import pickle
import numpy as np
from datetime import timezone
import nltk

# Download required NLTK data for processing language
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Set up OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Function to download PDFs from S3 prefix
def download_pdfs_from_s3_prefixes(bucket_name, prefixes, local_folder):
    s3 = boto3.client('s3')
    os.makedirs(local_folder, exist_ok=True)
    for prefix in prefixes:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.pdf'):
                filename = os.path.basename(key)
                local_path = os.path.join(local_folder, filename)
                s3_last_modified = obj['LastModified'].replace(tzinfo=timezone.utc).timestamp()
                if (not os.path.exists(local_path)) or (os.path.getmtime(local_path) < s3_last_modified):
                    s3.download_file(bucket_name, key, local_path)
                    print(f"Downloaded {key} to {local_path}")
                else:
                    print(f"Skipped {key}, local file is up to date.")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Clean text function
def clean_text(text):
    text = text.replace("\x00", "")
    text = text.encode("ascii", "ignore").decode()
    text = " ".join(text.split())
    # Convert to lowercase during cleaning
    text = text.lower()
    return text

# Split text into chunks
def split_text_into_chunks(text, max_tokens=500, overlap=100):
    # Ensure text is lowercase before splitting
    text = text.lower()
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

            overlap_words = chunk_text.split()[-overlap:]
            current_chunk = [' '.join(overlap_words), sentence]
            current_length = len(overlap_words) + sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Generate embeddings
def generate_embeddings(texts, batch_size=10):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=batch
            )
            embeddings.extend([embedding["embedding"] for embedding in response["data"]])
        except Exception as e:
            print(f"Error generating embeddings for batch {i}-{i+batch_size}: {e}")
    return embeddings

# Function to load existing document store from pickle file
def load_document_store(pickle_file_path):
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"No existing document store found at {pickle_file_path}. Creating new one.")
        return {}

# Function to process and append new PDFs from slack_pdfs to existing document store
def process_and_append_slack_pdfs(bucket_name, slack_prefix, local_pdf_folder, pickle_file_path):
    # Load existing document store from the specified path
    document_store = load_document_store(pickle_file_path)
    
    # Print document store info before processing
    print(f"Loaded document store with {len(document_store)} documents")
    print(f"Document keys: {list(document_store.keys())[:5]}...")
    
    # Download new PDFs from slack_pdfs
    download_pdfs_from_s3_prefixes(bucket_name, [slack_prefix], local_pdf_folder)
    
    # Process new PDFs and append to document_store
    new_docs_added = 0
    for filename in os.listdir(local_pdf_folder):
        if filename.endswith('.pdf'):
            file_path = os.path.join(local_pdf_folder, filename)
            # Check if file is already in document store
            if filename.lower() in document_store:
                print(f"Skipping {filename}, already in document store.")
                continue
                
            # Extract text from PDF
            text = extract_text_from_pdf(file_path)
            clean_doc_text = clean_text(text)
            text_chunks = split_text_into_chunks(clean_doc_text)
            
            if not text_chunks:
                print(f"Warning: No chunks generated for {filename}, skipping.")
                continue
                
            embeddings = generate_embeddings(text_chunks)
            
            if len(embeddings) != len(text_chunks):
                print(f"Warning: Mismatch between chunks and embeddings for {filename}. Got {len(embeddings)} embeddings for {len(text_chunks)} chunks.")
                text_chunks = text_chunks[:len(embeddings)]
            
            # Add source prefix to help with retrieval
            prefixed_chunks = [f"[SLACK PDF: {filename}] {chunk}" for chunk in text_chunks]
            
            # Append to document_store
            document_store[filename.lower()] = {
                "chunks": prefixed_chunks,
                "embeddings": embeddings,
                "source": "slack_pdf",
                "original_filename": filename
            }
            new_docs_added += 1
            print(f"Appended {filename} to the document store.")
    
    print(f"Added {new_docs_added} new documents to the store")
    
    # Save updated document_store back to pickle
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(document_store, f)
    print(f"Updated document store saved to {pickle_file_path}")
    
    return document_store

# Enhanced cosine similarity function with better numerical stability
def cosine_similarity(vec1, vec2):
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    if vec1_norm < 1e-10 or vec2_norm < 1e-10:
        return 0.0

    return np.dot(vec1, vec2) / (vec1_norm * vec2_norm)

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

# Improved function to retrieve relevant chunks with hybrid approach
def retrieve_relevant_chunks(query, document_store, top_k=5):
    query_embedding = generate_embeddings([query])[0]
    similarities = []
    
    # Extract key terms from query for keyword matching
    query_terms = set(term.lower() for term in query.split() 
                     if term.lower() not in {'how', 'what', 'when', 'where', 'do', 'does', 'is', 'are', 'the'})
    
    for doc_name, doc_data in document_store.items():
        for i, (chunk, chunk_embedding) in enumerate(zip(doc_data["chunks"], doc_data["embeddings"])):
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
    
    # Debug info
    if not relevant_chunks:
        print("Warning: No relevant chunks found for query:", query)
    else:
        print(f"Top similarity score: {relevant_chunks[0][1]:.4f} for document: {relevant_chunks[0][2]}")
    
    return [(chunk, doc_name) for chunk, _, doc_name in relevant_chunks]

# Function to chat with assistant
def chat_with_assistant(query, document_store):
    relevant_chunks = retrieve_relevant_chunks(query, document_store)
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
        max_tokens=1500
    )

    return response.choices[0].message.content.strip()

# Main execution
if __name__ == "__main__":
    # Configuration
    AWS_BUCKET_NAME = 'hcmbotknowledgesource'
    SLACK_PREFIX = 'slack_pdfs/'
    LOCAL_PDF_FOLDER = 'pdfs2'
    PICKLE_FILE_PATH = 'documen_store.pkl'  # Fixed to use the existing file
    
    # Process and append new PDFs from slack_pdfs to existing document store
    document_store = process_and_append_slack_pdfs(
        AWS_BUCKET_NAME, 
        SLACK_PREFIX, 
        LOCAL_PDF_FOLDER, 
        PICKLE_FILE_PATH
    )
    
    print(f"Document store now contains {len(document_store)} documents")
    
    # Optional: Test retrieval with a sample query
    test_query = "What information is in the slack PDFs?"
    relevant_chunks = retrieve_relevant_chunks(test_query, document_store)
    print(f"Found {len(relevant_chunks)} relevant chunks for test query")
