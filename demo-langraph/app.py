import streamlit as st
import os
import sys
import subprocess
from agent.graph import app as agent_app
from pathlib import Path

st.set_page_config(page_title="Syllabus RAG Agent", page_icon="", layout="wide")

st.title("Syllabus-Aware AI Assistant")

# Sidebar: file upload
with st.sidebar:
    st.header("Upload Syllabus")
    uploaded_file = st.file_uploader(
        "Upload a PDF syllabus",
        type=["pdf"],
        help="Upload your course syllabus PDF. It will be automatically processed."
    )
    
    if uploaded_file is not None:
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
        
        # Save file to data/uploads directory
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / uploaded_file.name
        file_key = f"{uploaded_file.name}_{uploaded_file.size}" 
        
        #  Save the uploaded file
        if file_key not in st.session_state.processed_files:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f" File saved: {uploaded_file.name}")
            
            # Automatic trigger ingestion
            with st.spinner("Processing document and creating embeddings"):
                try:
                    
                    python_executable = sys.executable
                    result = subprocess.run(
                        [python_executable, "ingestion/ingest.py"],
                        capture_output=True,
                        text=True,
                        timeout=300,  
                        cwd=os.getcwd()  
                    )
                    
                    if result.returncode == 0:
                        st.success(" Document processed successfully! You can now ask questions about it.")
                        st.session_state.processed_files.add(file_key) # Mark file as processed
                    else:
                        st.error(f" Error processing document:\n{result.stderr}")
                        if result.stdout:
                            st.text("Output:")
                            st.text(result.stdout)
                except subprocess.TimeoutExpired:
                    st.error("Processing timed out.")
                except Exception as e:
                    st.error(f" Error: {str(e)}")
        else:
            st.info(f" File '{uploaded_file.name}' was already processed in this session.")
        
        # Show file info
        if file_path.exists():
            file_size = os.path.getsize(file_path) / 1024
            st.info(f" File size: {file_size:.1f} KB")
    
    # Show existing uploaded files
    st.header(" Uploaded Files")
    upload_dir = Path("data/uploads")
    if upload_dir.exists():
        pdf_files = list(upload_dir.glob("*.pdf"))
        if pdf_files:
            for pdf_file in pdf_files:
                st.text(f"• {pdf_file.name}")
        else:
            st.info("No PDFs uploaded yet")
    else:
        st.info("Upload directory not created yet")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)

query = st.chat_input("Ask about your syllabus, weather (with date), or chat...")

if query:
    # Display user message immediately
    st.chat_message("user").write(query)
    st.session_state.messages.append(("user", query))
    
    # Format message history for the agent
    formatted_messages = []
    for role, msg in st.session_state.messages[:-1]: 
        formatted_messages.append(f"{role.capitalize()}: {msg}")
    
    #state
    state = {
        "query": query,
        "messages": formatted_messages,
    }
    
    # response from agent
    with st.spinner("Thinking"):
        try:
            result = agent_app.invoke(state)
            response = result.get("result", "I couldn't generate a response.")
        except Exception as e:
            response = f"Error: {str(e)}"
    
    #Display final except response from assitnat
    st.chat_message("assistant").write(response)
    st.session_state.messages.append(("assistant", response))

