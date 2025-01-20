#import neccessary libraries
#-----------------------------------------------------------------------------
import streamlit as st
import fitz 
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess

# Set page configuration
st.set_page_config(
    page_title="Multiple PDFs Local RAG System",
    page_icon="📄",
)


#STEP 1: Function to extract text from PDF
#-----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


# STEP 2/3: Embedding and Function to query the FAISS index
#-----------------------------------------------------------------------------
def query_faiss_index(query, index, embedding_model, metadata, k=3):
    query_embedding = embedding_model.encode([query])[0]
    faiss.normalize_L2(np.array([query_embedding]))
    distances, indices = index.search(np.array([query_embedding]), k)
    results = [(metadata[idx], dist) for idx, dist in zip(indices[0], distances[0])]
    return results


# STEP 4: Function to query Ollama
#-----------------------------------------------------------------------------
def query_ollama(model, prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error querying Ollama: {e.stderr.strip()}"


# STEP 5: Sync with Streamlit App
#-----------------------------------------------------------------------------
st.title("RAG System with Ollama and FAISS")
st.sidebar.header("Interactive Options")

# Sidebar for file uploads
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)

embedding_model_name = st.sidebar.selectbox(
    "Select Embedding Model",
    options=["all-MiniLM-L6-v2", "paraphrase-MiniLM-L12-v2", "multi-qa-MiniLM-L6-cos-v1"],
    index=0
)
chunk_size = st.sidebar.number_input("Chunk Size (words)", min_value=50, max_value=500, value=300, step=10)
top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=10, value=3)

# State initialization
if "index" not in st.session_state:
    st.session_state.index = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # To store user-bot chat history


#Upload section
#-----------------------------------------------------------------------------
if uploaded_files:
    st.sidebar.write("Uploaded files:")
    for file in uploaded_files:
        st.sidebar.write(f"- {file.name}")

    # File selection dropdown
    selected_file = st.sidebar.selectbox(
        "Select a file to process", [file.name for file in uploaded_files]
    )

    # Find the selected file
    file_to_process = next(
        (file for file in uploaded_files if file.name == selected_file), None
    )

    if file_to_process:
        st.write(f"##### Processing: {file_to_process.name}")

        # Extract text
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(file_to_process)

        # # Split text into chunks
        chunks = split_text_into_chunks(pdf_text, chunk_size=chunk_size)
        # st.write(f"Total Chunks: {len(chunks)}")

        with st.spinner("Loading embedding model..."):
            st.session_state.embedding_model = SentenceTransformer(embedding_model_name)

        with st.spinner("Generating embeddings and creating FAISS index..."):
            embeddings = st.session_state.embedding_model.encode(chunks)
            faiss.normalize_L2(np.array(embeddings))

            dimension = embeddings[0].shape[0]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))

            st.session_state.index = index
            st.session_state.metadata = {i: chunks[i] for i in range(len(chunks))}

        st.success("FAISS index created successfully!")


# Chat interface
#-----------------------------------------------------------------------------
if st.session_state.index:
    # Chat interface
    with st.container():
        for message in st.session_state.chat_history:
            if message["sender"] == "user":
                st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 10px;">
                    <div style="margin-right: 10px; background: #e0f7fa; color: #333; padding: 10px; border-radius: 5px; max-width: 80%;">
                        {message["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
                )
            else:
                st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 10px;">
                    <div style="background: #f1f8e9; color: #333; padding: 10px; border-radius: 5px; max-width: 80%;">
                        {message["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
                )

    # Process user query on Enter
    def process_query():
        user_query = st.session_state.get("query", "").strip()
        if not user_query:
            return  # Skip processing for empty queries

        with st.spinner("Searching FAISS index..."):
            faiss_results = query_faiss_index(
                user_query,
                st.session_state.index,
                st.session_state.embedding_model,
                st.session_state.metadata,
                k=top_k
            )

        retrieved_context = "\n\n".join([text for text, _ in faiss_results])

        refined_prompt = f"""Answer the following question based on the document context:
        Query: {user_query}
        Context: {retrieved_context}
        """
        with st.spinner("Generating response with Ollama..."):
            bot_response = query_ollama("llama3.2:latest", refined_prompt)

        # Update chat history
        st.session_state.chat_history.append({"sender": "user", "content": user_query})
        st.session_state.chat_history.append({"sender": "bot", "content": bot_response})

        # Clear query input field
        st.session_state["query"] = ""

    # Input form with real-time query processing on Enter
    st.text_input(
        "Ask a question:",
        key="query",
        on_change=process_query,
        placeholder="Type your query and press Enter",
    )


