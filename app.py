import os
import nltk
from pptx import Presentation
import openpyxl
import fitz  # PyMuPDF
import numpy as np
import openai

# Download required NLTK data for processing language
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Set base path to local folders
base_path = 'data'  # Change this to your desired local path
pdf_folder = os.path.join(base_path, "pdfs")
ppt_folder = os.path.join(base_path, "ppts")
excel_folder = os.path.join(base_path, "excels")

# Create directories if they don't exist
for folder in [pdf_folder, ppt_folder, excel_folder]:
    os.makedirs(folder, exist_ok=True)

# Authenticate OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Text Extraction Functions
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

# Step 2: Text Preprocessing and Chunking
def clean_text(text):
    text = text.replace("\x00", "")
    text = text.encode("ascii", "ignore").decode()
    text = " ".join(text.split())
    return text

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

# Store document chunks and their embeddings
document_store = {}

def process_and_upload_from_folder(folder_path, doc_type, chunking_strategy):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
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
            print(f"Processed {filename} ({doc_type.upper()}) successfully.")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

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
    print(f"Relevant chunks for query '{query}':")
    for chunk, doc in relevant_chunks:
        print(f"Source ({doc}): {chunk}")
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

def process_and_upload_all():
    print("Starting document processing...")
    process_and_upload_from_folder(pdf_folder, "pdf", split_text_into_chunks)
    process_and_upload_from_folder(ppt_folder, "ppt", split_text_into_chunks)
    process_and_upload_from_folder(excel_folder, "excel", split_text_into_chunks)
    print("Document processing complete.")

if __name__ == "__main__":
    # Process documents
    process_and_upload_all()

    # Test the system
    query = "what is the role of ministry of health in campaign digitization"
    answer = chat_with_assistant(query)
    print(f"Assistant's answer: {answer}")
