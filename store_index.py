from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
import os
import pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")  

if not PINECONE_API_KEY or not PINECONE_API_ENV:
    raise ValueError("PINECONE_API_KEY or PINECONE_API_ENVIRONMENT is missing from .env file.")

# Initialize Pinecone (v2.2.4 style)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Index name
index_name = "medical-bot"

# Check and create index if needed
if index_name not in pinecone.list_indexes():
    print(f"Index '{index_name}' not found. Creating it...")
    pinecone.create_index(
        name=index_name,
        dimension=384,  # All-MiniLM-L6-v2 => 384-dim embeddings
        metric="cosine"
    )
else:
    print(f"Index '{index_name}' already exists.")

# Get index instance (v2.2.4 style)
index = pinecone.Index(index_name)

# Prepare documents and embeddings
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Optional: limit for testing / chunk control
limited_chunks = text_chunks[:5000]

# Upload to Pinecone using LangChain
try:
    print("Uploading documents to Pinecone index...")
    docsearch = LangchainPinecone.from_texts(
        texts=[t.page_content for t in limited_chunks],
        embedding=embeddings,
        index=index,
        namespace="default"
    )
    print("Documents uploaded successfully.")
except Exception as e:
    print(f"Error occurred while uploading documents: {e}")

















