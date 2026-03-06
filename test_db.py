import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
client = chromadb.Client()

# Check if the collection exists, create it if it doesn't
collection_name = "policy_document_collection"

# Try to get the collection, create it if it doesn't exist
collection = client.get_collection(collection_name)
print(f"Collection '{collection_name}' exists.")
