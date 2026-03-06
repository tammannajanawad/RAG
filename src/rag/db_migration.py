import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Open the company policy document (assuming PDF format for this example)
with open("company_policy.pdf", "rb") as f:
    reader = PyPDF2.PdfReader(f)
    document_text = ""
    for page in reader.pages:
        document_text += page.extract_text()

# Function to split the document into chunks of 25 words
def split_into_chunks(text, chunk_size=25):
    words = text.split()  # Split text into words
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]  # Group into chunks of `chunk_size` words
    return [' '.join(chunk) for chunk in chunks]  # Join words back into string for each chunk

# Split the document into chunks of 25 words
chunks = split_into_chunks(document_text, chunk_size=25)

# Initialize ChromaDB client
client = chromadb.Client()

# Collection name
collection_name = "policy_document_collection"

# Create or get the collection
try:
    # Try to get the collection and add documents
    collection = client.get_collection(collection_name)
    print(f"Collection '{collection_name}' exists.")
except chromadb.errors.NotFoundError:
    print(f"Collection '{collection_name}' does not exist. Creating now.")
    collection = client.create_collection(name=collection_name)
    print(f"Collection '{collection_name}' created successfully.")

# Embed the chunks and add them to the collection
for idx, chunk in enumerate(chunks):
    embedding = model.encode(chunk)
    
    # Provide unique IDs for each chunk
    collection.add(
        documents=[chunk],  # The chunk of text
        ids=[f"chunk_{idx}"],  # Unique ID for each chunk (e.g., "chunk_0", "chunk_1", etc.)
        metadatas=[{'chunk_id': idx}],  # Metadata to track the chunk ID
        embeddings=[embedding]  # The embedded vector
    )

print("Document chunks embedded and stored in ChromaDB successfully.")

# Query the collection to verify it has documents
query_results = collection.query(
    query_embeddings=[model.encode("test query")],  # Query with a random test embedding
    n_results=5  # Retrieve the top 5 results
)

print(f"Collection '{collection_name}' is now accessible and contains {len(query_results['documents'])} documents.")
