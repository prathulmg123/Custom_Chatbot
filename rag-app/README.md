# RAG Document Q&A Application

This is a full-stack application that allows you to upload documents and ask questions about their content using Retrieval-Augmented Generation (RAG) with Ollama's LLM and ChromaDB for vector storage.

## Features

- Upload PDF or text documents
- Ask questions about the uploaded documents
- View sources for the answers
- Modern, responsive UI built with React and Chakra UI
- FastAPI backend with LangChain and ChromaDB

## Prerequisites

- Python 3.8+
- Node.js 16+
- Ollama installed and running locally (for the LLM)

## Setup

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

   In a separate terminal, pull the model:
   ```bash
   ollama pull llama3
   ```

5. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
   The server will start at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install the required Node.js packages:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:5173`

## Usage

1. Open the application in your browser at `http://localhost:5173`
2. Upload a PDF or text document using the upload form
3. Once uploaded, ask questions about the document content
4. The application will provide answers based on the document content

## Project Structure

```
rag-app/
├── backend/
│   ├── main.py           # FastAPI application
│   ├── requirements.txt   # Python dependencies
│   ├── data/             # Uploaded documents
│   └── chroma_db/        # Vector store (created at runtime)
└── frontend/
    ├── src/
    │   ├── App.jsx       # Main React component
    │   └── main.jsx      # React entry point
    ├── package.json      # Node.js dependencies
    └── index.html        # HTML template
```

## Troubleshooting

- If you encounter CORS issues, make sure the backend is running and the CORS middleware is properly configured in `main.py`
- If the LLM is not responding, ensure Ollama is running and the model is downloaded
- Check the browser's developer console for any frontend errors
- Check the backend logs for any server-side errors

## License

MIT
