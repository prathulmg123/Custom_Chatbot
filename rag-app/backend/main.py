import os
import shutil
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.llms import Ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
DATA_DIR = str(Path(__file__).parent / "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="RAG Q&A: LangChain + Chroma + HuggingFace",
    description="Document Q&A with RAG using LangChain, Chroma, and HuggingFace"
)

# CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to cache the embeddings model
_embeddings = None

def get_embeddings():
    global _embeddings
    
    # Return cached embeddings if available
    if _embeddings is not None:
        logger.debug("Using cached embeddings")
        return _embeddings
        
    try:
        logger.info("Initializing HuggingFace embeddings...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        logger.info(f"Loading model: {model_name}")
        try:
            # Initialize with minimal configuration first
            _embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "show_progress_bar": False,
                    "batch_size": 32
                }
            )
            logger.info("Successfully initialized embeddings with standard configuration")
        except Exception as e:
            logger.warning(f"Standard initialization failed, trying with minimal configuration: {str(e)}")
            _embeddings = HuggingFaceEmbeddings(model_name=model_name)
            logger.info("Successfully initialized embeddings with minimal configuration")
        
        # Test the embeddings with a simple query
        logger.info("Testing embeddings with a sample query...")
        try:
            test_embedding = _embeddings.embed_query("test")
            if not test_embedding or len(test_embedding) == 0:
                raise ValueError("Generated empty embedding")
            logger.info(f"Successfully tested embeddings. Vector dimensions: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"Embedding test failed: {str(e)}")
            _embeddings = None
            raise
        
        return _embeddings
        
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}", exc_info=True)
        try:
            logger.info("Trying with minimal configuration...")
            _embeddings = HuggingFaceEmbeddings(model_name=model_name)
            test_embedding = _embeddings.embed_query("test")
            if test_embedding and len(test_embedding) > 0:
                logger.info("Successfully initialized with minimal configuration")
                return _embeddings
        except Exception as fallback_error:
            logger.error(f"Fallback initialization also failed: {str(fallback_error)}")
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize embeddings: {str(e)}"
        )

def get_vectorstore():
    try:
        logger.info("Getting embeddings for vector store...")
        embeddings = get_embeddings()
        
        # Ensure the directory exists and has correct permissions
        os.makedirs(PERSIST_DIR, exist_ok=True)
        os.chmod(PERSIST_DIR, 0o755)  # Ensure directory is writable
        
        logger.info(f"Initializing Chroma vector store at {PERSIST_DIR}...")
        
        try:
            # Try with collection_metadata first
            vs = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings,
                collection_name="documents",
                collection_metadata={"hnsw:space": "cosine"}
            )
            logger.info("Successfully initialized Chroma with collection_metadata")
        except Exception as e:
            logger.warning(f"Initialization with collection_metadata failed, trying without: {str(e)}")
            vs = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings,
                collection_name="documents"
            )
            logger.info("Successfully initialized Chroma with default settings")
        
        # Verify collection exists
        collection = vs._collection
        if collection is None:
            raise ValueError("Failed to create or access collection in Chroma")
            
        logger.info(f"Vector store initialized successfully. Collection: {collection.name}")
        return vs
        
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}", exc_info=True)
        # Try to clean up and retry once
        try:
            if os.path.exists(PERSIST_DIR):
                logger.info(f"Attempting to clean up and retry initialization. Removing: {PERSIST_DIR}")
                shutil.rmtree(PERSIST_DIR)
                os.makedirs(PERSIST_DIR, exist_ok=True)
                os.chmod(PERSIST_DIR, 0o755)
                
                logger.info("Retrying vector store initialization...")
                vs = Chroma(
                    persist_directory=PERSIST_DIR,
                    embedding_function=embeddings,
                    collection_name="documents"
                )
                logger.info("Successfully reinitialized vector store after cleanup")
                return vs
        except Exception as cleanup_error:
            logger.error(f"Cleanup and retry failed: {str(cleanup_error)}")
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize vector store: {str(e)}"
        )

def load_file_to_docs(path: str):
    try:
        logger.info(f"Loading file: {path}")
        file_size = os.path.getsize(path)
        logger.info(f"File size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError("Uploaded file is empty")
            
        ext = Path(path).suffix.lower()
        logger.info(f"File extension: {ext}")
        
        try:
            if ext == ".pdf":
                logger.info("Using PyPDFLoader")
                loader = PyPDFLoader(path)
            else:
                logger.info("Using TextLoader")
                # Try different encodings if UTF-8 fails
                try:
                    loader = TextLoader(path, encoding="utf-8")
                except UnicodeDecodeError:
                    logger.info("UTF-8 failed, trying ISO-8859-1")
                    loader = TextLoader(path, encoding="ISO-8859-1")
            
            # Load and validate documents
            docs = loader.load()
            if not docs:
                raise ValueError("No documents found in the file")
                
            logger.info(f"Loaded {len(docs)} document(s)")
            
            # Add source metadata and validate content
            for i, doc in enumerate(docs):
                if not doc.page_content.strip():
                    logger.warning(f"Document {i+1} has empty content")
                    continue
                    
                doc.metadata["source"] = os.path.basename(path)
                logger.info(f"Document {i+1} metadata: {doc.metadata}")
                logger.info(f"Document {i+1} content preview: {doc.page_content[:200]}...")
            
            # Filter out any empty documents
            valid_docs = [doc for doc in docs if doc.page_content.strip()]
            if not valid_docs:
                logger.error("All documents had empty content after stripping whitespace")
                raise ValueError("No valid content found in the document")
                
            return valid_docs
            
        except Exception as e:
            # Try alternative loaders if the primary one fails
            if "PDF" in str(e) and ext == ".pdf":
                logger.warning("PDF loading failed, trying alternative approach")
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(path)
                    text = "\n\n".join([page.extract_text() for page in reader.pages])
                    if not text.strip():
                        raise ValueError("Could not extract any text from PDF")
                    from langchain.schema import Document
                    return [Document(page_content=text, metadata={"source": os.path.basename(path)})]
                except Exception as pdf_err:
                    logger.error(f"Alternative PDF loading also failed: {str(pdf_err)}")
                    raise ValueError(f"Failed to process PDF: {str(pdf_err)}")
            raise
            
    except Exception as e:
        error_msg = f"Error loading file {path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process document: {str(e)}"
        )

def split_docs(docs):
    try:
        # Optimize the text splitter for better performance
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,  # Reduced overlap for better performance
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],  # More efficient splitting
            keep_separator=True
        )
        
        # Process documents in parallel if there are many
        if len(docs) > 1:
            import concurrent.futures
            
            def process_doc(doc):
                return splitter.split_documents([doc])
                
            with concurrent.futures.ThreadPoolExecutor() as executor:
                chunks = []
                for doc_chunks in executor.map(process_doc, docs):
                    chunks.extend(doc_chunks)
        else:
            chunks = splitter.split_documents(docs)
        
        if not chunks:
            raise ValueError("No chunks were generated from the document")
            
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to split document: {str(e)}"
        )

def configure_gemini(api_key: str = None):
    """Configure the Gemini model with API key."""
    try:
        if not api_key:
            raise ValueError("Google API key is required")
            
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure Gemini: {str(e)}"
        )

def get_llm():
    try:
        model_name = "mistral"  # Using Mistral through Ollama
        base_url = "http://localhost:11434"  # Default Ollama server URL
        
        logger.info(f"Initializing Ollama with model: {model_name}")
        
        # Initialize Ollama LLM
        llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            num_ctx=2048  # Context window size
        )
        
        # Test the model
        test_response = llm.invoke("Test connection")
        if not test_response:
            raise ValueError("Failed to get response from Ollama server")
            
        logger.info("Successfully connected to Ollama server")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to initialize Ollama with Mistral: {str(e)}", exc_info=True)
        error_msg = (
            "Failed to connect to Ollama server. "
            "Please ensure Ollama is running and the Mistral model is downloaded.\n"
            "1. Install Ollama: https://ollama.ai/\n"
            "2. Run: ollama pull mistral\n"
            "3. Start Ollama server: ollama serve"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

@app.post("/chat")
async def chat(question: str):
    try:
        logger.info(f"Received question: {question}")
        
        # Get vector store and retriever
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.7,
            },
        )
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(question)
        
        if not docs:
            return {
                "answer": "I couldn't find enough relevant information to answer your question.",
                "sources": []
            }
        
        # Format context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set([doc.metadata["source"] for doc in docs]))
        
        # Get Ollama LLM
        llm = get_llm()
        
        # Create prompt for Mistral
        prompt = f"""[INST] <<SYS>>
        You are a helpful AI assistant that answers questions based on the provided context.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        <</SYS>>
        
        Context:
        {context}
        
        Question: {question}
        
        Answer: [/INST]"""
        
        # Generate response
        answer = llm.invoke(prompt)
        
        # Log response
        logger.info(f"Generated response: {answer}")
        
        return {
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing your question: {str(e)}"
        )

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers using ONLY the provided context.
If the answer is not in the context, say you don't know.

Question: {input}

Context:
{context}

Answer (include brief citations like [source]):"""
)

class QueryIn(BaseModel):
    query: str
    top_k: Optional[int] = 4

class QueryOut(BaseModel):
    answer: str
    sources: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_added: int
    file: str
    total_documents_in_db: int

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    start_time = time.time()
    temp_file_path = None
    
    try:
        logger.info(f"Starting file upload: {file.filename}")
        
        # Validate file type and size (max 50MB)
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF and TXT files are supported"
            )
        
        # Create a temporary file with a unique name
        temp_file_path = f"temp_{int(time.time())}_{file.filename}"
        temp_file_path = str(Path(DATA_DIR) / temp_file_path)
        
        # Save uploaded file in chunks to handle large files
        logger.info(f"Saving file to temporary location: {temp_file_path}")
        with open(temp_file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                f.write(chunk)
        
        file_size = os.path.getsize(temp_file_path) / (1024 * 1024)  # Size in MB
        logger.info(f"File saved successfully. Size: {file_size:.2f} MB")
        
        # Load and process document
        logger.info("Loading and processing document...")
        load_start = time.time()
        try:
            docs = load_file_to_docs(temp_file_path)
            logger.info(f"Document loaded in {time.time() - load_start:.2f} seconds")
            
            # Split into chunks
            logger.info("Splitting document into chunks...")
            split_start = time.time()
            chunks = split_docs(docs)
            logger.info(f"Document split into {len(chunks)} chunks in {time.time() - split_start:.2f} seconds")
            
            if not chunks:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid content could be extracted from the document"
                )
            
            # Initialize vector store
            logger.info("Initializing vector store...")
            vs_start = time.time()
            vs = get_vectorstore()
            logger.info(f"Vector store initialized in {time.time() - vs_start:.2f} seconds")
            
            # Add documents in smaller batches to reduce memory usage
            batch_size = 8  # Reduced batch size
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            logger.info(f"Starting to process {len(chunks)} chunks in {total_batches} batches...")
            add_start = time.time()
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                batch_start = time.time()
                
                try:
                    vs.add_documents(batch)
                    logger.info(
                        f"Processed batch {batch_num}/{total_batches} "
                        f"(chunks {i+1}-{min(i+batch_size, len(chunks))}) "
                        f"in {time.time() - batch_start:.2f}s"
                    )
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {str(e)}")
                    raise
            
            # Persist the vector store
            logger.info("Persisting vector store...")
            persist_start = time.time()
            vs.persist()
            logger.info(f"Vector store persisted in {time.time() - persist_start:.2f}s")
            
            # Verify documents were added
            total_docs = vs._collection.count() if hasattr(vs, '_collection') else len(chunks)
            logger.info(f"Added {total_docs} documents in {time.time() - add_start:.2f} seconds")
            
            total_time = time.time() - start_time
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            
            return {
                "status": "success",
                "message": f"Document processed in {total_time:.2f} seconds",
                "chunks_added": len(chunks),
                "file": file.filename,
                "total_documents_in_db": total_docs,
                "processing_time_seconds": round(total_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Error during document processing: {str(e)}", exc_info=True)
            # Clean up the vector store to avoid partial data
            if os.path.exists(PERSIST_DIR):
                logger.info("Cleaning up partial vector store...")
                try:
                    shutil.rmtree(PERSIST_DIR)
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up vector store: {str(cleanup_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process document: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ingest: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")

@app.post("/query", response_model=QueryOut)
def query(q: QueryIn):
    try:
        vs = get_vectorstore()
        retriever = vs.as_retriever(search_kwargs={"k": q.top_k})

        llm = get_llm()
        combine_chain = create_stuff_documents_chain(llm, PROMPT)
        rag_chain = create_retrieval_chain(retriever, combine_chain)

        result = rag_chain.invoke({"input": q.query})
        
        # Collect unique sources
        sources = []
        for doc in result.get("context", []):
            source = doc.metadata.get("source")
            if source and source not in sources:
                sources.append(source)
                
        return {
            "answer": result["answer"],
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
