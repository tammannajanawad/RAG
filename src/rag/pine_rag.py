import os
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import requests
import json

# Configuration
CHROMA_DB_PATH = "./chroma_db"  # Local directory for ChromaDB storage
COLLECTION_NAME = "policy-documents"
PDF_PATH = "company_policy.pdf"
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"  # Default LM Studio endpoint

# Initialize embedding model (using all-MiniLM-L6-v2 for lightweight embeddings)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_chroma_collection():
    """Create or get ChromaDB collection"""
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Get or create collection
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists")
    except:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Policy documents collection"}
        )
        print(f"Created collection: {COLLECTION_NAME}")
    
    return collection

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF and split into chunks"""
    reader = PdfReader(pdf_path)
    chunks = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        # Split into smaller chunks (you can adjust chunk size)
        chunk_size = 60
        words = text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append({
                    'text': chunk,
                    'page': page_num + 1,
                    'chunk_id': f"page_{page_num + 1}_chunk_{i // chunk_size}"
                })
    
    print(f"Extracted {len(chunks)} chunks from PDF")
    return chunks

def embed_and_store(collection, chunks):
    """Generate embeddings and store in ChromaDB"""
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        # Generate embedding
        embedding = embedding_model.encode(chunk['text']).tolist()
        
        # Prepare data for ChromaDB
        ids.append(chunk['chunk_id'])
        embeddings.append(embedding)
        documents.append(chunk['text'])
        metadatas.append({'page': chunk['page']})
    
    # Add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"Uploaded all {len(chunks)} vectors to ChromaDB")

def query_deepseek(prompt, context):
    """Query DeepSeek model via LM Studio"""
    full_prompt = f"""Based on the following context from policy documents, please answer the question.

Context:
{context}

Question: {prompt}

Answer:"""
    
    payload = {
        "model": "deepseek",  # Adjust if your model has a different name
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(LM_STUDIO_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error querying DeepSeek: {str(e)}"

def search_and_answer(collection, query, top_k=3):
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
    print("\n--- Retrieved Contexts ---")
    
    if results['documents'] and len(results['documents'][0]) > 0:
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i] if results['distances'] else 0
            page = results['metadatas'][0][i]['page'] if results['metadatas'] else 'N/A'
            
            # ChromaDB returns distance (lower is better), convert to similarity score
            similarity = 1 - distance
            
            print(f"Score: {similarity:.4f} | Page: {page}")
            print(f"Text: {doc[:200]}...\n")
            contexts.append(doc)
    
    combined_context = "\n\n".join(contexts)
    
    # Get answer from DeepSeek
    print("--- Generating Answer ---")
    answer = query_deepseek(query, combined_context)
    
    return answer

def main():
    """Main function to run the RAG system"""
    print("=== RAG System with ChromaDB and DeepSeek ===\n")
    
    # Step 1: Create collection
    print("Step 1: Setting up ChromaDB collection...")
    collection = create_chroma_collection()
    
    # Step 2: Process PDF (only do this once)
    print("\nStep 2: Processing PDF document...")
    
    # Check if collection already has documents
    count = collection.count()
    if count > 0:
        print(f"Collection already has {count} documents")
        process_pdf = input("Do you want to re-process and upload the PDF? (yes/no): ").lower()
    else:
        process_pdf = input("Do you want to process and upload the PDF? (yes/no): ").lower()
    
    if process_pdf == 'yes':
        if not os.path.exists(PDF_PATH):
            print(f"Error: PDF file not found at {PDF_PATH}")
            return
        
        # Clear existing collection if re-processing
        if count > 0:
            print("Clearing existing collection...")
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            client.delete_collection(name=COLLECTION_NAME)
            collection = create_chroma_collection()
        
        chunks = extract_text_from_pdf(PDF_PATH)
        embed_and_store(collection, chunks)
        print("PDF processing complete!")
    
    # Step 3: Query loop
    print("\n=== Ready for Questions ===")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        answer = search_and_answer(collection, query)
        print(f"\nAnswer: {answer}\n")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    main()