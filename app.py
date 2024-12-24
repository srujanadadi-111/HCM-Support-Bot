import streamlit as st
import openai
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
    "https://drive.google.com/uc?id=1VR9AppuVbuli0d_8_VMP83sXW6GxBq4v"
]

# Helper function to create the vector store from the PDFs
def upload_pdfs_to_vector_store(client, vector_store_id, pdf_files):
    try:
        file_ids = {}
        for file_url in pdf_files:
            try:
                # Download the file and upload it to the vector store
                uploaded_file = client.beta.vector_stores.files.upload(
                    vector_store_id=vector_store_id, 
                    file_url=file_url
                )
                file_ids[file_url] = uploaded_file.id
            except Exception as file_error:
                st.error(f"Error uploading file {file_url}: {file_error}")
        
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
            instructions="You are an expert assistant. Respond only based on the documents provided.",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
            temperature=0.0
        )
        return assistant
    except Exception as e:
        st.error(f"Error creating assistant: {e}")
        return None

# Streamlit UI
st.title("HCM Support Bot")

# Input: Vector Store Name
vector_store_name = st.text_input("Enter Vector Store Name:", "DocumentResearchStore")

# Upload PDF files to the vector store
vector_store = get_or_create_vector_store(client, vector_store_name)
if vector_store:
    file_ids = upload_pdfs_to_vector_store(client, vector_store.id, pdf_files)
    if file_ids:
        st.success(f"Uploaded {len(file_ids)} files successfully.")
    else:
        st.warning("No files uploaded.")
else:
    st.error("Failed to create or retrieve vector store.")

# Assistant Interaction
st.subheader("Chat with the Assistant")
assistant_query = st.text_area("Enter your question:")

if st.button("Ask"):
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
