import os
import pickle
import boto3
import fitz  # PyMuPDF
import nltk
import numpy as np
import openai
from datetime import datetime, timezone

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Get credentials from environment variables
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Set OpenAI API key
openai.api_key = openai_api_key

# Constants
AWS_BUCKET_NAME = 'hcmbotknowledgesource'
LOCAL_PDF_FOLDER = 'pdfs2'
BASE_PKL_FILE = 'documen_store_og.pkl'
FINAL_PKL_FILE = 'document_store.pkl'

# Text processing functions
def clean_text(text):
    text = text.replace("\x00", "")
    text = text.encode("ascii", "ignore").decode()
    text = " ".join(text.split())
    text = text.lower()
    return text

def split_text_into_chunks(text, max_tokens=500, overlap=100):
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

# S3 Functions
def get_s3_client():
    """Create and return an S3 client using environment variables"""
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

def download_latest_slack_pdf():
    """Download the most recent slack PDF from S3"""
    s3 = get_s3_client()
    
    # List objects in the slack_pdfs prefix
    response = s3.list_objects_v2(
        Bucket=AWS_BUCKET_NAME,
        Prefix='slack_pdfs/'
    )
    
    if 'Contents' not in response:
        print("No slack PDFs found in S3")
        return None
    
    # Find the most recent file
    latest_file = None
    latest_date = None
    
    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith('.pdf'):
            # Extract date from filename (assuming format hcmsupportbot_YYYY-MM-DD.pdf)
            try:
                date_str = key.split('_')[1].replace('.pdf', '')
                file_date = datetime.strptime(date_str, '%Y-%m-%d')
                
                if latest_date is None or file_date > latest_date:
                    latest_date = file_date
                    latest_file = key
            except (IndexError, ValueError):
                continue
    
    if latest_file:
        # Create directory if it doesn't exist
        os.makedirs(LOCAL_PDF_FOLDER, exist_ok=True)
        
        # Download the file
        local_path = os.path.join(LOCAL_PDF_FOLDER, os.path.basename(latest_file))
        s3.download_file(AWS_BUCKET_NAME, latest_file, local_path)
        print(f"Downloaded {latest_file} to {local_path}")
        return local_path
    
    return None

def download_base_pdfs():
    """Download base PDFs from S3"""
    s3 = get_s3_client()
    os.makedirs(LOCAL_PDF_FOLDER, exist_ok=True)
    
    # List objects in the base_pdfs prefix
    response = s3.list_objects_v2(
        Bucket=AWS_BUCKET_NAME,
        Prefix='base_pdfs/'
    )
    
    if 'Contents' not in response:
        print("No base PDFs found in S3")
        return []
    
    downloaded_files = []
    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith('.pdf'):
            filename = os.path.basename(key)
            local_path = os.path.join(LOCAL_PDF_FOLDER, filename)
            s3_last_modified = obj['LastModified'].replace(tzinfo=timezone.utc).timestamp()
            
            if (not os.path.exists(local_path)) or (os.path.getmtime(local_path) < s3_last_modified):
                s3.download_file(AWS_BUCKET_NAME, key, local_path)
                print(f"Downloaded {key} to {local_path}")
                downloaded_files.append(local_path)
            else:
                print(f"Skipped {key}, local file is up to date.")
                downloaded_files.append(local_path)
    
    return downloaded_files

# PDF Processing
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_pdf(pdf_path):
    """Process a single PDF and return its chunks and embeddings"""
    print(f"Processing PDF: {pdf_path}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    # Clean and chunk
    clean_doc_text = clean_text(text)
    text_chunks = split_text_into_chunks(clean_doc_text)
    
    # Generate embeddings
    embeddings = generate_embeddings(text_chunks)
    
    # Return processed data
    filename = os.path.basename(pdf_path).lower()
    return {
        filename: {
            "chunks": text_chunks[:len(embeddings)],
            "embeddings": embeddings,
            "source": "slack_pdf" if "hcmsupportbot" in filename else "base_pdf",
            "original_filename": os.path.basename(pdf_path)
        }
    }

def load_or_create_document_store():
    """Load existing document store or create a new one"""
    try:
        # Try to download the base document store from S3
        s3 = get_s3_client()
        try:
            s3.download_file(AWS_BUCKET_NAME, BASE_PKL_FILE, BASE_PKL_FILE)
            print(f"Downloaded base document store from S3")
            
            with open(BASE_PKL_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Could not download base document store: {e}")
            
            # Try to load locally
            if os.path.exists(BASE_PKL_FILE):
                with open(BASE_PKL_FILE, 'rb') as f:
                    return pickle.load(f)
    except Exception as e:
        print(f"Error loading document store: {e}")
    
    # Create new if not found
    return {}

def main():
    # Load existing document store
    document_store = load_or_create_document_store()
    print(f"Loaded document store with {len(document_store)} documents")
    
    # Download and process the latest slack PDF
    latest_slack_pdf = download_latest_slack_pdf()
    
    if latest_slack_pdf:
        # Process the latest slack PDF
        slack_data = process_pdf(latest_slack_pdf)
        
        # Update the document store with new data
        document_store.update(slack_data)
        print(f"Added latest slack PDF to document store")
        
        # Save the updated document store
        with open(FINAL_PKL_FILE, 'wb') as f:
            pickle.dump(document_store, f)
        print(f"Saved updated document store to {FINAL_PKL_FILE}")
        
        # Upload to S3
        s3 = get_s3_client()
        s3.upload_file(FINAL_PKL_FILE, AWS_BUCKET_NAME, FINAL_PKL_FILE)
        print(f"Uploaded document store to S3")
    else:
        print("No new slack PDF to process")

if __name__ == "__main__":
    main()
