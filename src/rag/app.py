import streamlit as st
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import requests
import json
import os
import tempfile
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="RAG Chat with PDF",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file, chunk_size):
    """Extract text from PDF and split into chunks"""
    reader = PdfReader(pdf_file)
    chunks = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        words = text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append({
                    'text': chunk,
                    'page': page_num + 1,
                    'chunk_id': f"page_{page_num + 1}_chunk_{i // chunk_size}"
                })
    
    return chunks

def create_chroma_collection(db_path, collection_name):
    """Create or get ChromaDB collection"""
    client = chromadb.PersistentClient(path=db_path)
    
    # Delete collection if it exists (for fresh start)
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "PDF documents collection"}
    )
    
    return collection

def embed_and_store(collection, chunks, embedding_model):
    """Generate embeddings and store in ChromaDB"""
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, chunk in enumerate(chunks):
        # Generate embedding
        embedding = embedding_model.encode(chunk['text']).tolist()
        
        # Prepare data for ChromaDB
        ids.append(chunk['chunk_id'])
        embeddings.append(embedding)
        documents.append(chunk['text'])
        metadatas.append({'page': chunk['page']})
        
        # Update progress
        progress = (idx + 1) / len(chunks)
        progress_bar.progress(progress)
        status_text.text(f"Processing chunk {idx + 1}/{len(chunks)}")
    
    # Add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    progress_bar.empty()
    status_text.empty()
    
    return len(chunks)

def query_deepseek(prompt, context, lm_studio_url, temperature, max_tokens):
    """Query DeepSeek model via LM Studio"""
    full_prompt = f"""Based on the following context from the document, please answer the question.

Context:
{context}

Question: {prompt}

Answer:"""
    
    payload = {
        "model": "deepseek",
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(lm_studio_url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        return "Error: Request timed out. LM Studio might be processing a large request."
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to LM Studio. Please ensure it's running on the specified URL."
    except Exception as e:
        return f"Error querying DeepSeek: {str(e)}"

def search_and_answer(collection, query, embedding_model, top_k, lm_studio_url, temperature, max_tokens, show_sources):
    """Search ChromaDB and generate answer using DeepSeek"""
    # Generate query embedding
    query_embedding = embedding_model.encode(query).tolist()
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Combine retrieved contexts
    contexts = []
    sources_info = []
    
    if results['documents'] and len(results['documents'][0]) > 0:
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i] if results['distances'] else 0
            page = results['metadatas'][0][i]['page'] if results['metadatas'] else 'N/A'
            similarity = 1 - distance
            
            contexts.append(doc)
            sources_info.append({
                'page': page,
                'similarity': similarity,
                'text': doc
            })
    
    combined_context = "\n\n".join(contexts)
    
    # Get answer from DeepSeek
    answer = query_deepseek(query, combined_context, lm_studio_url, temperature, max_tokens)
    
    return answer, sources_info

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.header("📄 PDF Upload")
    uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
    
    st.header("🗄️ Database Settings")
    db_path = st.text_input("Database Path", value="./chroma_db")
    collection_name = st.text_input("Collection Name", value=f"pdf_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    st.header("✂️ Processing Settings")
    chunk_size = st.slider("Chunk Size (words)", min_value=20, max_value=1000, value=500, step=20)
    top_k = st.slider("Number of Results to Retrieve", min_value=1, max_value=10, value=3)
    
    st.header("🤖 LM Studio Settings")
    lm_studio_url = st.text_input("LM Studio URL", value="http://localhost:1234/v1/chat/completions")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=2000, value=500, step=100)
    
    st.header("📊 Display Settings")
    show_sources = st.checkbox("Show Source Contexts", value=True)
    
    st.divider()
    
    # Process PDF button
    if uploaded_file is not None:
        if st.button("🚀 Process PDF", type="primary", use_container_width=True):
            with st.spinner("Processing PDF..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    # Extract chunks
                    st.info(f"Extracting text from PDF...")
                    chunks = extract_text_from_pdf(tmp_path, chunk_size)
                    st.success(f"Extracted {len(chunks)} chunks from PDF")
                    
                    # Create collection
                    st.info("Creating database collection...")
                    collection = create_chroma_collection(db_path, collection_name)
                    
                    # Embed and store
                    st.info("Generating embeddings and storing in database...")
                    num_stored = embed_and_store(collection, chunks, st.session_state.embedding_model)
                    
                    # Update session state
                    st.session_state.collection = collection
                    st.session_state.pdf_processed = True
                    st.session_state.messages = []  # Clear chat history
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    st.success(f"✅ Successfully processed and stored {num_stored} chunks!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
    
    # Clear chat button
    if st.session_state.pdf_processed:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Main content area
st.title("📚 RAG Chat with PDF Documents")
st.markdown("Upload a PDF, configure settings, and chat with your document!")

# Check if PDF is processed
if not st.session_state.pdf_processed:
    st.info("👈 Please upload a PDF and click 'Process PDF' to get started.")
else:
    st.success("✅ PDF processed! Ask questions below.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message and show_sources:
                with st.expander("📖 View Source Contexts"):
                    for idx, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {idx + 1}** (Page {source['page']}, Similarity: {source['similarity']:.4f})")
                        st.text(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = search_and_answer(
                        st.session_state.collection,
                        prompt,
                        st.session_state.embedding_model,
                        top_k,
                        lm_studio_url,
                        temperature,
                        max_tokens,
                        show_sources
                    )
                    
                    st.markdown(answer)
                    
                    # Show sources
                    if show_sources and sources:
                        with st.expander("📖 View Source Contexts"):
                            for idx, source in enumerate(sources):
                                st.markdown(f"**Source {idx + 1}** (Page {source['page']}, Similarity: {source['similarity']:.4f})")
                                st.text(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])
                                st.divider()
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>RAG System with ChromaDB and DeepSeek | Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)