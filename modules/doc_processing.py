import os
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,UnstructuredFileLoader # type: ignore
# from google.generativeai import embed_text
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))

# from log.logger import setup_logger
# logger = setup_logger("data_utils_logs")

base_dir = os.path.dirname(os.path.abspath(__file__)) 

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Paths for directories
UPLOAD_DIR = os.path.join(base_dir, "..","data")
VECTOR_DB_PATH = os.path.join(base_dir, "..","data", "vector_store.faiss")


def load_data():
    """Load data from the uploaded files in the specified directory."""
    if not os.path.exists(UPLOAD_DIR):
        # logger.error(f"The directory '{UPLOAD_DIR}' does not exist.")
        return None
    
    if not os.listdir(UPLOAD_DIR):
        # logger.warning(f"The directory '{UPLOAD_DIR}' is empty.")
        return None

    try:      
        # Load PDF files
        pdf_loader = DirectoryLoader(
            UPLOAD_DIR, 
            glob="*.pdf", 
            loader_cls=PyPDFLoader
        )

        # Load DOCX files
        docx_loader = DirectoryLoader(
            UPLOAD_DIR, 
            glob="*.docx", 
            loader_cls=UnstructuredFileLoader
        )
        
        documents = documents = pdf_loader.load() + docx_loader.load()

        # print(documents)
        if not documents:
            # logger.warning(f"No documents found in the directory '{UPLOAD_DIR}' that match the specified formats.")
            return None
        
        # logger.info(f"Successfully loaded {len(documents)} documents.")
        return documents

    except Exception as e:
        # logger.error(f"An error occurred while loading documents: {e}")
        return None



def split_data(documents):
    """Split documents into smaller chunks using a text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=150)
    return text_splitter.split_documents(documents)


def save_embeddings(text_chunks):
    """Generate embeddings using Google Generative AI and save to a FAISS vector database."""
    # Use Google Generative AI for embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Create FAISS database from the text chunks
    vector_db = FAISS.from_documents(text_chunks, embeddings)
    
    # Save FAISS database to the specified path
    vector_db.save_local(VECTOR_DB_PATH)
    # logger.info(f"FAISS vector database saved to {VECTOR_DB_PATH}")
