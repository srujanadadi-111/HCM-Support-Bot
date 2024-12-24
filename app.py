import streamlit as st
import os
import openai
import time

#st.write("Secrets keys available:", list(st.secrets.keys()))


# Fetch the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]

if not openai_api_key:
    st.error("OpenAI API key not found. Please configure secrets.")
else:
    st.success("OpenAI API key loaded successfully.")

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

MODEL_NAME = "gpt-3.5-turbo"

# Helper functions for uploading PDFs, creating vector store, etc.
def upload_pdfs_to_vector_store(client, vector_store_id, directory_path):
    try:
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Error: Directory '{directory_path}' does not exist.")
        
        pdf_files = [
            os.path.join(directory_path, file) 
            for file in os.listdir(directory_path) 
            if file.lower().endswith(".pdf")
        ]
        
        if not pdf_files:
            st.warning("No PDF files found in directory.")
            return {}

        file_ids = {}
        for file_path in pdf_files:
            try:
                with open(file_path, "rb") as file:
                    uploaded_file = client.beta.vector_stores.files.upload(
                        vector_store_id=vector_store_id, 
                        file=file
                    )
                file_ids[os.path.basename(file_path)] = uploaded_file.id
            except Exception as file_error:
                st.error(f"Error uploading {file_path}: {file_error}")
        
        return file_ids

    except Exception as e:
        st.error(f"Error in file upload process: {e}")
        return {}

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

def create_assistant(client, model_name, vector_store_id):
    try:
        assistant = client.beta.assistants.create(
            model=model_name,
            name="Document Research Assistant",
            description="Assistant for searching and analyzing PDF documents.",
            instructions="You are an expert assistant. Respond only based on the documents provided.",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
            temperature=0.0
        )
        return assistant
    except Exception as e:
        st.error(f"Error creating assistant: {e}")
        return None

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

# Input: Vector Store Name
vector_store_name = st.text_input("Enter Vector Store Name:", "DocumentResearchStore")

# File Upload Section
st.subheader("Upload PDFs")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if st.button("Upload"):
    vector_store = get_or_create_vector_store(client, vector_store_name)
    if vector_store:
        if uploaded_files:
            upload_dir = "./uploads"
            os.makedirs(upload_dir, exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
            file_ids = upload_pdfs_to_vector_store(client, vector_store.id, upload_dir)
            if file_ids:
                st.success(f"Uploaded {len(file_ids)} files successfully.")
            else:
                st.warning("No files uploaded.")
        else:
            st.warning("Please upload at least one PDF.")
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
