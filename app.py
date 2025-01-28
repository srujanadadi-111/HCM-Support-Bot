import os
import sys
import nltk
from pptx import Presentation
import openpyxl
import fitz  # PyMuPDF
import numpy as np
import openai
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download required NLTK data for processing language
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
except Exception as e:
    logging.error(f"Failed to download NLTK data: {str(e)}")
    sys.exit(1)

# Set base path to local folders
base_path = 'data'  # Change this to your desired local path
pdf_folder = os.path.join(base_path, "pdfs")
ppt_folder = os.path.join(base_path, "ppts")
excel_folder = os.path.join(base_path, "excels")

# Create directories if they don't exist
try:
    for folder in [pdf_folder, ppt_folder, excel_folder]:
        os.makedirs(folder, exist_ok=True)
        logging.info(f"Created/verified folder: {folder}")
except Exception as e:
    logging.error(f"Failed to create directories: {str(e)}")
    sys.exit(1)

# Authenticate OpenAI API
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    logging.info("OpenAI API key loaded successfully")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI: {str(e)}")
    sys.exit(1)

def process_and_upload_all():
    logging.info("Starting document processing...")
    try:
        # Check if any documents exist
        has_documents = False
        for folder in [pdf_folder, ppt_folder, excel_folder]:
            if os.path.exists(folder) and any(os.listdir(folder)):
                has_documents = True
                break
        
        if not has_documents:
            logging.warning("No documents found in any folder. Please add documents first.")
            return False

        process_and_upload_from_folder(pdf_folder, "pdf", split_text_into_chunks)
        process_and_upload_from_folder(ppt_folder, "ppt", split_text_into_chunks)
        process_and_upload_from_folder(excel_folder, "excel", split_text_into_chunks)
        
        if not document_store:
            logging.warning("No documents were successfully processed.")
            return False
            
        logging.info("Document processing complete.")
        return True
    except Exception as e:
        logging.error(f"Error during document processing: {str(e)}")
        return False

def chat_with_assistant(query):
    try:
        if not document_store:
            return "No documents have been processed yet. Please add documents and run processing first."
            
        relevant_chunks = retrieve_relevant_chunks(query)
        if not relevant_chunks:
            return "No relevant information found for your query."

        logging.info(f"Processing query: {query}")
        context = "\n\n".join([f"Source ({doc}): {chunk}" for chunk, doc in relevant_chunks])

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": """You are a precise assistant that answers questions based strictly on the provided context."""
                }, {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }],
                temperature=0,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {str(e)}")
            return "Sorry, I encountered an error while processing your question. Please try again later."
            
    except Exception as e:
        logging.error(f"Error in chat_with_assistant: {str(e)}")
        return "An error occurred while processing your question."

if __name__ == "__main__":
    try:
        logging.info("Starting application...")
        if process_and_upload_all():
            logging.info("Document processing successful, starting query loop...")
            while True:
                try:
                    query = input("Enter your question (or 'quit' to exit): ")
                    if query.lower() == 'quit':
                        break
                        
                    answer = chat_with_assistant(query)
                    if answer:
                        print(f"\nAnswer: {answer}\n")
                except KeyboardInterrupt:
                    logging.info("Received keyboard interrupt, exiting...")
                    break
                except Exception as e:
                    logging.error(f"Error processing query: {str(e)}")
        else:
            logging.error("Failed to process documents. Check the logs above.")
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        sys.exit(1)
