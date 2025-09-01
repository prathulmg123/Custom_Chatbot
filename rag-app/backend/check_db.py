from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

# Configuration
PERSIST_DIR = str(Path(__file__).parent / "chroma_db")

def check_database():
    try:
        # Initialize Chroma with the same settings as in main.py
        embeddings = FastEmbedEmbeddings()
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        
        # Get the collection
        collection = db._collection
        if not collection:
            print("No collection found in the database.")
            return
            
        # Count documents
        count = collection.count()
        print(f"Number of documents in the database: {count}")
        
        # Get some sample documents if they exist
        if count > 0:
            print("\nSample documents:")
            results = collection.get(include=["documents", "metadatas"])
            for i, (doc, meta) in enumerate(zip(results['documents'][:3], results['metadatas'][:3])):
                print(f"\nDocument {i+1}:")
                print(f"Content: {doc[:200]}...")
                print(f"Metadata: {meta}")
        
    except Exception as e:
        print(f"Error checking database: {str(e)}")

if __name__ == "__main__":
    check_database()
