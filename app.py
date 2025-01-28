import streamlit as st
import openai
import requests
import os
import time
import nltk
import fitz  # PyMuPDF
import pinecone
from tqdm import tqdm

# Fetch API keys from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]
pinecone_api_key = st.secrets["pinecone_api_key"]
pinecone_environment = st.secrets["pinecone_environment"]  # e.g., "gcp-starter"
index_name = "document-store"  # Your Pinecone index name

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

MODEL_NAME = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Hardcoded Google Drive PDF file links
pdf_files = [
    "https://drive.google.com/uc?id=1VR9AppuVbuli0d_8_VMP83sXW6GxBq4v",
    "https://drive.google.com/uc?id=1ET7BcoCts75yevG76tgWxzbrspF9MeZ9",
    "https://drive.google.com/uc?id=1OIBuluuqEuM4AMRxVRLBlOeax5IjkQF0",
    "https://drive.google.com/uc?id=15bBXXAoYHi0pc57ZZmo8UqhNjlZRHjrm",
    "https://drive.google.com/uc?id=1obv8-kbBqX6Ucp7ciRP4ZOZ-9keq1OlJ",
    "https://drive.google.com/uc?id=1vrHy5tX2h65l6cC_PW-vDDHdO5r1KHST"
]

@st.cache_resource
def initialize_pinecone():
    """Initialize Pinecone client and create index if it doesn't exist"""
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    
    # Check if index exists
    if index_name not in pinecone.list_indexes():
        # Create index with desired configuration
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536  # dimension for ada-002 embeddings
        )
    
    return pinecone.Index(index_name)

def clean_text(text):
    """Clean and preprocess text"""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())
    return text

def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def generate_embedding(text):
    """Generate a single embedding using OpenAI's API"""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def download_pdf_from_drive(file_url, save_path):
    """Download PDF from Google Drive"""
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
    """Process PDF file and upload chunks to Pinecone"""
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
        filename = os.path.basename(file_path)
        vectors_to_upsert = []
        
        for i, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)
            if embedding:
                vector_id = f"{filename}_chunk_{i}"
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "source": filename
                    }
                })
        
        # Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            pinecone_index.upsert(vectors=batch)
        
        return True
    except Exception as e:
        st.error(f"Error processing file {file_path}: {e}")
        return False

def query_pinecone(query, pinecone_index, top_k=3):
    """Query Pinecone index for similar chunks"""
    try:
        query_embedding = generate_embedding(query)
        if query_embedding:
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            return [(match.metadata["text"], match.metadata["source"]) for match in results.matches]
        return []
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return []

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
    # Get existing vector IDs
    existing_ids = set()
    try:
        # Fetch all vector IDs (you might need to implement pagination for large datasets)
        stats = pinecone_index.describe_index_stats()
        if stats.total_vector_count == 0:
            st.info("No documents found in database. Starting initial load...")
            
            for pdf_file in pdf_files:
                with st.spinner(f'Processing {pdf_file.split("=")[-1]}...'):
                    file_name = pdf_file.split("=")[-1] + ".pdf"
                    file_path = os.path.join("./downloads", file_name)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    downloaded_file = download_pdf_from_drive(pdf_file, file_path)
                    if downloaded_file and process_and_upload_to_pinecone(downloaded_file, pinecone_index):
                        st.success(f"Successfully processed {file_name}")
            
            st.success("All documents loaded successfully!")
        else:
            st.success(f"Documents already loaded! Found {stats.total_vector_count} vectors in database.")
    except Exception as e:
        st.error(f"Error checking document status: {e}")

# Chat interface
st.subheader("Ask Questions")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching for answer..."):
        relevant_chunks = query_pinecone(query, pinecone_index)
        
        # Display relevant chunks in an expander
        with st.expander("View source chunks"):
            for chunk, doc_name in relevant_chunks:
                st.markdown(f"**Source**: {doc_name}")
                st.write(chunk)
                st.markdown("---")
        
        # Generate response using ChatGPT
        if relevant_chunks:
            context = "\n\n".join([f"Source ({doc}): {chunk}" for chunk, doc in relevant_chunks])
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
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
                    temperature=0
                )
                st.write("Answer:", response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error generating response: {e}")
        else:
            st.warning("No relevant information found in the documents.")
