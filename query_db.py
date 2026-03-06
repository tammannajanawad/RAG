import chromadb
from sentence_transformers import SentenceTransformer
import requests

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
client = chromadb.Client()

# Collection name
collection_name = "policy_document_collection"

# Try to get the collection, create if it doesn't exist
try:
    collection = client.get_collection(collection_name)
    print(f"Collection '{collection_name}' exists.")
except chromadb.errors.NotFoundError:
    print(f"Collection '{collection_name}' does not exist. Creating now.")
    collection = client.create_collection(name=collection_name)
    print(f"Collection '{collection_name}' created successfully.")

# Local DeepSeek API URL (change this to your actual local URL)
local_deepseek_url = "http://localhost:1234/api/v1/chat"

# Function to embed the query and retrieve relevant documents
def retrieve_documents(query, top_k=3):
    # Create an embedding for the query
    query_embedding = model.encode(query)
    
    # Query ChromaDB for top-k relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    print(results)
    # Extract the retrieved documents from the query results
    # Each result will contain 'documents' as a list of relevant documents
    retrieved_docs = [doc[0] if isinstance(doc, list) else doc for doc in results['documents']]
    
    return retrieved_docs

# Function to generate an answer with the locally hosted model
def generate_answer_with_deepseek(retrieved_docs, query):
    # Ensure that the documents are strings (extract if necessary)
    context = "\n".join([str(doc) for doc in retrieved_docs])  # Convert all documents to strings
    
    # Construct the prompt
    prompt = f"Based on the following company policy document, answer the question:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Prepare the request data
    data = {
        "prompt": prompt,
        "max_tokens": 150
    }

    # Send the request to the local server (DeepSeek)
    response = requests.post(local_deepseek_url, json=data)

    if response.status_code == 200:
        answer = response.json().get("generated_text", "No answer generated.")
        return answer
    else:
        return f"Error: {response.status_code} - {response.text}"

# Example query
query = "What are the company's policies on remote work?"

# Retrieve relevant documents from ChromaDB
retrieved_docs = retrieve_documents(query)

# Get the generated answer using the locally hosted DeepSeek model
answer = generate_answer_with_deepseek(retrieved_docs, query)
print(f"Generated Answer: {answer}")
