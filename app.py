import streamlit as st
import openai
import requests
import os
import fitz  # PyMuPDF
from tqdm import tqdm
import pinecone

# Fetch API keys from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]
pinecone_api_key = st.secrets["pinecone_api_key"]
pinecone_environment = st.secrets["pinecone_environment"]  # e.g., "us-west1-gcp"
index_name = "document-store"  # Your Pinecone index name

# Initialize OpenAI
openai.api_key = openai_api_key

MODEL_NAME = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Hardcoded Google Drive PDF file links
pdf_files = [
    "https://drive.google.com/uc?id=1VR9AppuVbuli0d_8_VMP83sXW6GxBq4v",
    "https://drive.google.com/uc?id=1ET7BcoCts75yevG76tgWxzbrspF9MeZ9",
    # Add more files as needed
]

@st.cache_resource
def initialize_pinecone():
    """Initialize Pinecone and return the index object."""
    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    
    # Check if index exists; create it if it doesn't
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # Dimension for Ada embeddings
            metric="cosine"
        )
    
    # Return the index object
    return pinecone.Index(index_name)

def clean_text(text):
    """Clean and preprocess text."""
    return ' '.join(text.replace('\n', ' ').replace('\r', ' ').split())

def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

def generate_embedding(text):
    """Generate a single embedding using OpenAI's API."""
    try:
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def download_pdf_from_drive(file_url, save_path):
    """Download PDF from Google Drive."""
    try:
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return save_path
        else:
            st.error(f"Failed to download file from {file_url}")
            return None
    except Exception as e:
        st.error(f"Error downloading file: {e}")
        return None

def process_and_upload_to_pinecone(file_path, pinecone_index):
    """Process PDF file and upload chunks to Pinecone."""
    try:
        # Extract text from PDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Process text
        clean_doc_text = clean_text(text)
        chunks = split_text_into_chunks(clean_doc_text)
        
        # Generate embeddings and upload to Pinecone
        vectors_to_upsert = []
        filename = os.path.basename(file_path)
        
        for i, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)
            if embedding:
                vectors_to_upsert.append({
                    "id": f"{filename}_chunk_{i}",
                    "values": embedding,
                    "metadata": {"text": chunk, "source": filename}
                })
        
        # Upsert to Pinecone in batches
        pinecone_index.upsert(vectors=vectors_to_upsert)
        return True
    except Exception as e:
        st.error(f"Error processing file {file_path}: {e}")
        return False

# Streamlit UI
st.title("HCM Support Bot")

# Initialize Pinecone
try:
    pinecone_index = initialize_pinecone()
except Exception as e:
    st.error(f"Error connecting to Pinecone: {e}")
    st.stop()

# Document loading interface
if st.button("Check and Load Documents"):
    try:
        stats = pinecone_index.describe_index_stats()
        if stats["total_vector_count"] == 0:
            st.info("No documents found in the database. Starting initial load...")
            
            for pdf_file in pdf_files:
                file_name = pdf_file.split("=")[-1] + ".pdf"
                file_path = os.path.join("./downloads", file_name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                downloaded_file = download_pdf_from_drive(pdf_file, file_path)
                if downloaded_file and process_and_upload_to_pinecone(downloaded_file, pinecone_index):
                    st.success(f"Successfully processed {file_name}")
            
            st.success("All documents loaded successfully!")
        else:
            st.success(f"Documents already loaded! Found {stats['total_vector_count']} vectors in database.")
    except Exception as e:
        st.error(f"Error checking document status: {e}")

# Chat interface
st.subheader("Ask Questions")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching for answer..."):
        try:
            query_embedding = generate_embedding(query)
            if query_embedding:
                results = pinecone_index.query(
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True
                )
                if results.matches:
                    for match in results.matches:
                        st.write(f"Source: {match['metadata']['source']}")
                        st.write(match['metadata']['text'])
                else:
                    st.warning("No matching results found.")
        except Exception as e:
            st.error(f"Error querying Pinecone: {e}")
