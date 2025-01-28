import os
import openai
import fitz  # PyMuPDF
#from pptx import Presentation
import openpyxl
import streamlit as st
import gdown
import numpy as np
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_TOKENS = 500  # Adjust based on your use case

# Google Drive folder link
pdf_folder_url = "https://drive.google.com/drive/folders/1YJZlxPM_pGAae7IRllMCr_mxvkquBWdB?usp=sharing"

# Functions to download files from Google Drive
def download_files_from_drive(folder_url, save_path):
    """Downloads all files in a Google Drive folder to the specified path."""
    file_ids = gdown.download_folder(folder_url)
    for file_id in file_ids:
        save_file_path = os.path.join(save_path, file_id.split('/')[-1])
        gdown.download(file_id, save_file_path, quiet=False)
    return file_ids

# Extract text functions
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_ppt(ppt_path):
    prs = Presentation(ppt_path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def extract_text_from_excel(excel_path):
    wb = openpyxl.load_workbook(excel_path)
    sheet = wb.active
    text = ""
    for row in sheet.iter_rows():
        for cell in row:
            text += str(cell.value) + "\n"
    return text

# Text processing functions
def clean_text(text):
    text = text.replace("\x00", "")
    text = text.encode("ascii", "ignore").decode()
    text = " ".join(text.split())
    return text

def split_text_into_chunks(text, max_tokens=MAX_TOKENS, overlap=100):
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

# Embedding function
def generate_embeddings(texts, batch_size=10):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = openai.Embedding.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            embeddings.extend([embedding["embedding"] for embedding in response["data"]])
        except Exception as e:
            st.error(f"Error generating embeddings for batch {i}-{i+batch_size}: {e}")
    return embeddings

# Process and store documents in memory
document_store = {}

def process_and_upload_from_folder(folder_path, doc_type):
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

        clean_doc_text = clean_text(text)
        text_chunks = split_text_into_chunks(clean_doc_text)
        embeddings = generate_embeddings(text_chunks)

        document_store[filename] = {
            "chunks": text_chunks[:len(embeddings)],
            "embeddings": embeddings,
            "source": doc_type
        }
        st.success(f"Uploaded {filename} ({doc_type.upper()}) to memory store.")

# Query and retrieval function
def cosine_similarity(vec1, vec2):
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    if vec1_norm == 0 or vec2_norm == 0:
        return 0.0
    return np.dot(vec1, vec2) / (vec1_norm * vec2_norm)

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
    context = "\n\n".join([f"Source ({doc}): {chunk}" for chunk, doc in relevant_chunks])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": """You are a precise assistant that answers questions based strictly on the provided context."""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("Document Processor and Chat Assistant")

# Upload documents button
if st.button("Process Documents from Google Drive"):
    # Download files from Google Drive
    download_files_from_drive(pdf_folder_url, save_path="./downloads")
    process_and_upload_from_folder("./downloads", "pdf")
    st.success("Documents processed successfully!")

# Chat interface
query = st.text_input("Ask a Question:")

if query:
    with st.spinner("Fetching response..."):
        try:
            answer = chat_with_assistant(query)
            st.write(f"Answer: {answer}")
        except Exception as e:
            st.error(f"Error: {e}")
