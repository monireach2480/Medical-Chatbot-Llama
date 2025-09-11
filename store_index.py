from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Extract data
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Get embedding dimension dynamically
dim = len(embeddings.embed_query("hello world"))

# Create index if it does not exist
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"✅ Created new index: {index_name}")
else:
    print(f"ℹ️ Index '{index_name}' already exists. Using existing one.")

# Load existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Add new docs to the index by uncommenting this line
docsearch.add_texts([t.page_content for t in text_chunks])
print("✅ Documents added to the Pinecone index successfully.")