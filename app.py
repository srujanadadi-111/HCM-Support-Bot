import streamlit as st
import openai
import requests
import os
import time

# Fetch the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]

if not openai_api_key:
    st.error("OpenAI API key not found. Please configure secrets.")
else:
    st.success("OpenAI API key loaded successfully.")

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

MODEL_NAME = "gpt-3.5-turbo"

# Hardcoded Google Drive PDF file links
pdf_files = [
    "https://drive.google.com/uc?id=1VR9AppuVbuli0d_8_VMP83sXW6GxBq4v",
    "https://drive.google.com/uc?id=1ET7BcoCts75yevG76tgWxzbrspF9MeZ9",
    "https://drive.google.com/uc?id=1OIBuluuqEuM4AMRxVRLBlOeax5IjkQF0",
    "https://drive.google.com/uc?id=15bBXXAoYHi0pc57ZZmo8UqhNjlZRHjrm",
    "https://drive.google.com/uc?id=1obv8-kbBqX6Ucp7ciRP4ZOZ-9keq1OlJ",
    "https://drive.google.com/uc?id=1vrHy5tX2h65l6cC_PW-vDDHdO5r1KHST"
    
    
]

# Helper function to download a PDF from Google Drive
def download_pdf_from_drive(file_url, save_path):
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

# Helper function to upload a PDF to the vector store
def upload_pdfs_to_vector_store(client, vector_store_id, pdf_file_paths):
    try:
        file_ids = {}
        for file_path in pdf_file_paths:
            try:
                with open(file_path, "rb") as file:
                    uploaded_file = client.beta.vector_stores.files.upload(
                        vector_store_id=vector_store_id, 
                        file=file
                    )
                file_ids[file_path] = uploaded_file.id
            except Exception as file_error:
                st.error(f"Error uploading file {file_path}: {file_error}")
        
        return file_ids

    except Exception as e:
        st.error(f"Error in file upload process: {e}")
        return {}

# Function to create or retrieve a vector store
def get_or_create_vector_store(client, vector_store_name):
    try:
        vector_stores = client.beta.vector_stores.list()
        for store in vector_stores.data:
            if store.name == vector_store_name:
                return store
        return client.beta.vector_stores.create(name=vector_store_name)
    except Exception as e:
        st.error(f"Error managing vector store: {e}")
        return None

# Function to create an assistant for querying the PDFs
def create_assistant(client, model_name, vector_store_id):
    try:
        assistant = client.beta.assistants.create(
            model=model_name,
            name="Document Research Assistant",
            description="Assistant for searching and analyzing PDF documents.",
            instructions="You are an expert assistant designed to help users with minimal exposure to digital technology. Assume that the user has little to no familiarity with digital devices, including phones or computers. Your primary role is to provide clear, in-depth answers strictly based on the referenced documents. Do not include any information that is not directly supported by the documents provided. Explain concepts and instructions thoroughly, using simple and relatable language without compromising on accuracy or terminology. Avoid brevity; provide detailed, step-by-step explanations to ensure the user fully understands the topic. Be empathetic and patient, recognizing the user's unfamiliarity with technology. Anticipate areas where they might need extra guidance and proactively address these with examples or additional explanations. Additionally, ask leading follow-up questions to guide the user toward a deeper understanding or uncover more specific needs related to their query. Stay focused, supportive, and committed to helping the user achieve their goals.",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
            temperature=0.0
        )
        return assistant
    except Exception as e:
        st.error(f"Error creating assistant: {e}")
        return None

# Function to wait for the completion of the run
def wait_for_run_completion(client, thread_id, run_id, max_attempts=500, delay=5):
    for attempt in range(max_attempts):
        try:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if run.status == 'completed':
                return run.status
            if run.status == 'failed':
                st.error("Run Failed. Check error details.")
                return run.status
            time.sleep(delay)
        except Exception as e:
            st.error(f"Error checking run status: {e}")
            return 'error'
    return 'timeout'

# Streamlit UI
st.title("HCM Support Bot")

# Vector Store Name (this can be hardcoded as well)
vector_store_name = "DocumentResearchStore"

# Automatically download PDFs from Google Drive and save locally
downloaded_files = []
vector_store = get_or_create_vector_store(client, vector_store_name)

if vector_store is not None:
    for pdf_file in pdf_files:
        file_name = pdf_file.split("=")[-1] + ".pdf"  # Derive file name from the ID
        file_path = os.path.join("./downloads", file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        downloaded_file = download_pdf_from_drive(pdf_file, file_path)
        if downloaded_file:
            downloaded_files.append(downloaded_file)

    # Now upload the downloaded files to the vector store
    if downloaded_files:
        file_ids = upload_pdfs_to_vector_store(client, vector_store.id, downloaded_files)
        if file_ids:
            st.success(f"Uploaded {len(file_ids)} files successfully.")
        else:
            st.warning("No files uploaded.")
    else:
        st.warning("No PDFs were downloaded.")
else:
    st.error("Failed to create or retrieve vector store.")

# Assistant Interaction
st.subheader("Chat with the Assistant")
assistant_query = st.text_area("Enter your question:")

if st.button("Ask"):
    vector_store = get_or_create_vector_store(client, vector_store_name)
    if vector_store:
        assistant = create_assistant(client, MODEL_NAME, vector_store.id)
        if assistant and assistant_query.strip():
            try:
                thread = client.beta.threads.create(
                    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
                )
                message = client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=assistant_query
                )
                run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
                
                # Wait for completion and get the response
                run_status = wait_for_run_completion(client, thread.id, run.id)
                if run_status == 'completed':
                    messages = client.beta.threads.messages.list(
                        thread_id=thread.id,
                        order='asc'
                    )
                    for msg in reversed(messages.data):
                        if msg.role == 'assistant':
                            st.success("Assistant Response: " + msg.content[0].text.value)
                            break
                else:
                    st.error(f"Run did not complete successfully. Status: {run_status}")
            except Exception as e:
                st.error(f"Error during chat: {e}")
        else:
            st.warning("Assistant creation failed or query is empty.")
    else:
        st.error("Failed to retrieve or create vector store.")
