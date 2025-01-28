import os
import gdown
import openai
import nltk
import numpy as np
from pptx import Presentation
import openpyxl
import fitz  # PyMuPDF
import streamlit as st

# Install required packages (if not installed already)
# !pip install gdown openai nltk pymupdf python-pptx openpyxl streamlit

# Setup OpenAI API Key
openai.api_key = openai_api_key  # Set your OpenAI API key here

# Download required NLTK data for processing language:
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Google Drive File ID (replace with actual file ID from the Google Drive URL)
file_id = "1BfgdaqBgQL-ku0l-iCQTYA4i5WKQw32V"  # Replace with your actual file ID
download_url = f"https://drive.google.com/uc?id={file_id}"

# Folder to save the downloaded files
save_folder = "./downloads"

# Ensure the folder exists
os.makedirs(save_folder, exist_ok=True)

# Define the file path to save the file
save_file_path = os.path.join(save_folder, "downloaded_file.pdf")  # Adjust file name and extension as needed

# Download the file
gdown.download(download_url, save_file_path, quiet=False)

st.write(f"File downloaded to {save_file_path}")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract text from PPT
def extract_text_from_ppt(ppt_path):
    prs = Presentation(ppt_path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

# Extract text from Excel
def extract_text_from_excel(excel_path):
    wb = openpyxl.load_workbook(excel_path)
    sheet = wb.active
    text = ""
    for row in sheet.iter_rows():
        for cell in row:
            text += str(cell.value) + "\n"
    return text

# Clean text to remove unwanted characters
def clean_text(text):
    text = text.replace("\x00", "")
    text = text.encode("ascii", "ignore").decode()
    text = " ".join(text.split())
    return text

# Split text into smaller chunks
def split_text_into_chunks(text, max_tokens=500, overlap=100):
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            # Create chunk with overlap
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

            # Keep last few sentences for overlap
            overlap_words = chunk_text.split()[-overlap:]
            current_chunk = [' '.join(overlap_words), sentence]
            current_length = len(overlap_words) + sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Generate embeddings for text using OpenAI
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

# Cosine similarity function for comparing embeddings
def cosine_similarity(vec1, vec2):
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    if vec1_norm == 0 or vec2_norm == 0:
        return 0.0

    return np.dot(vec1, vec2) / (vec1_norm * vec2_norm)

# Retrieve relevant chunks for a query based on cosine similarity
def retrieve_relevant_chunks(query, document_store, top_k=3):
    query_embedding = generate_embeddings([query])[0]
    similarities = []

    for doc_name, doc_data in document_store.items():
        for chunk, chunk_embedding in zip(doc_data["chunks"], doc_data["embeddings"]):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk, similarity, doc_name))

    relevant_chunks = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunk, doc_name) for chunk, _, doc_name in relevant_chunks]

# Document store to keep track of document chunks and their embeddings
document_store = {}

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
    relevant_chunks = retrieve_relevant_chunks(query, document_store)
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
def process_and_upload_all():
    process_and_upload_from_folder(save_folder, "pdf", split_text_into_chunks)
    process_and_upload_from_folder(save_folder, "ppt", split_text_into_chunks)
    process_and_upload_from_folder(save_folder, "excel", split_text_into_chunks)

# Streamlit interface
st.title("Document Processing and Chat Assistant")

# Button to start processing
if st.button('Process Documents and Upload'):
    process_and_upload_all()
    st.write("Documents processed and uploaded.")

# User input for querying
query = st.text_input("Ask a question:")
if query:
    answer = chat_with_assistant(query)
    st.write(f"Assistant's answer: {answer}")
