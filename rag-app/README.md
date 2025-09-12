# RAG Document Q&A Application

A powerful full-stack application that enables document-based question answering using Retrieval-Augmented Generation (RAG) with Ollama's LLM and ChromaDB for efficient vector storage and retrieval.

## ✨ Features

- **Document Upload**: Support for PDF and text document uploads
- **Intelligent Q&A**: Ask natural language questions about your documents
- **Source Citation**: View the exact sources used to generate each answer
- **Modern UI**: Clean, responsive interface built with React and Material-UI
- **Powerful Backend**: FastAPI server with LangChain integration for RAG pipeline
- **Efficient Storage**: ChromaDB for fast vector similarity search
- **Local LLM**: Utilizes Mistral through Ollama for private, offline processing

## 🚀 Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- Ollama (for local LLM processing)
- pip (Python package manager)
- npm or yarn (Node.js package manager)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-app
```

### 2. Backend Setup

1. Navigate to the backend directory and set up a virtual environment:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Ollama service in a separate terminal:
   ```bash
   ollama serve
   ```

4. (In a new terminal) Pull the required LLM model (Mistral):
   ```bash
   ollama pull mistral
   ```

5. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`
   - API documentation: `http://localhost:8000/docs`
   - Interactive API docs: `http://localhost:8000/redoc`

### 3. Frontend Setup

1. In a new terminal, navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server (it will automatically open in your default browser):
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:3000`

## 🎯 Usage

1. **Access the Application**: The application should automatically open in your default browser at `http://localhost:3000`. If it doesn't, you can manually navigate to this URL.
2. **Upload Documents**:
   - Click the upload button
   - Select one or more PDF or text files
   - Wait for the upload and processing to complete
3. **Ask Questions**:
   - Type your question in the chat interface
   - Press Enter or click the send button
   - View the AI-generated response with source citations
4. **View Sources**:
   - Each response includes references to the source documents
   - Click on the source links to view the relevant sections

## 🏗️ Project Structure

```
rag-app/
├── backend/
│   ├── main.py           # FastAPI application with all endpoints
│   ├── requirements.txt   # Python dependencies
│   ├── check_db.py       # Utility for database inspection
│   ├── data/             # Directory for uploaded documents
│   │   └── ...           # Uploaded files are stored here
│   └── chroma_db/        # ChromaDB vector store (auto-created)
│       └── ...           # Vector embeddings and metadata
└── frontend/
    ├── src/
    │   ├── App.jsx       # Main React component
    │   ├── main.jsx      # React entry point with providers
    │   └── index.css     # Global styles
    ├── public/           # Static assets
    ├── package.json      # Node.js dependencies
    └── index.html        # HTML template
```

## 🔧 Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Ensure the backend server is running
   - Check that the frontend is making requests to the correct port (default: 8000)
   - Verify CORS settings in `backend/main.py`

2. **LLM Not Responding**:
   - Make sure Ollama is running (`ollama serve`)
   - Verify the Mistral model is downloaded (`ollama pull mistral`)
   - Check the terminal where Ollama is running for errors

3. **Document Processing Issues**:
   - Ensure documents are not password protected
   - Check that the `data` directory has write permissions
   - Verify the document format is supported (PDF or plain text)

4. **Frontend Issues**:
   - Clear browser cache if UI is not updating
   - Check browser's developer console for errors (F12)
   - Ensure all Node.js dependencies are installed

### Checking Database Status

Use the included `check_db.py` script to inspect the vector database:

```bash
cd backend
python check_db.py


Issues Faced and Solutions


1. Layout Issues
   - Problem: Image and text not aligned
   - Fix: Used `st.columns()` for side-by-side layout

2. Image Upload
   - Problem: Only certain image formats worked
   - Fix: Added support for JPG, JPEG, PNG

3. API Key Security
   - Problem: Hardcoded API key
   - Note: For production, use environment variables


Future Improvements
    • Add support for batch image processing
    • Implement user authentication
    • Add more customization options for the AI model
    • Include error handling for API rate limits
    • Add a feature to save analysis results
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue in the repository.
