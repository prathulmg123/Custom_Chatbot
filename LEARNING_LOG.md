# Learning Log: RAG Application Development

## Project Overview
Building a Retrieval-Augmented Generation (RAG) application with a React frontend and FastAPI backend for document-based question answering.

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd rag-app/backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables (create a `.env` file):
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```
5. Start the backend server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup
1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd rag-app/frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```
3. Start the development server:
   ```bash
   npm start
   # or
   yarn start
   ```
4. The application should open automatically in your default browser at `http://localhost:3000`

### Accessing the Application
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Topics Learned

### 1. React Frontend Development
- **State Management**: Learned to use React hooks (useState) for managing component state
- **Conditional Rendering**: Implemented dynamic UI updates based on application state
- **Material-UI Components**: Gained experience with various MUI components for building responsive UIs
- **Form Handling**: Managed form inputs and file uploads in React
- **API Integration**: Connected frontend with backend API endpoints using Axios

### 2. FastAPI Backend
- **API Development**: Created RESTful endpoints for document processing
- **File Handling**: Implemented file upload and processing logic
- **Vector Database**: Integrated with a vector database for semantic search
- **Asynchronous Processing**: Used async/await for non-blocking operations

### 3. RAG Implementation
- **Document Processing**: Learned about text extraction and chunking
- **Embeddings**: Understanding of text embeddings for semantic search
- **Retrieval Mechanisms**: Implemented document retrieval based on semantic similarity
- **Response Generation**: Integrated with language models for generating answers

## Issues Faced & Solutions

### 1. Conditional UI Rendering
**Issue**: Needed to show/hide UI sections based on file upload status
**Solution**: Implemented state management with `useState` to track upload status and conditionally render components
```javascript
const [isFileUploaded, setIsFileUploaded] = useState(false);
// After successful upload:
setIsFileUploaded(true);
```

### 2. File Upload Handling
**Issue**: Managing file state and validation
**Solution**: Created proper file handling with validation and user feedback
```javascript
const handleFileChange = (e) => {
  const selectedFile = e.target.files[0];
  if (selectedFile && (selectedFile.type === 'application/pdf' || selectedFile.type === 'text/plain')) {
    setFile(selectedFile);
  } else {
    toast.error('Please upload a valid file (PDF or text)');
  }
};
```

### 3. State Management
**Issue**: Managing multiple related states
**Solution**: Consolidated related states and used functional updates when needed
```javascript
const clearAll = () => {
  setQuestion('');
  setAnswer('');
  setSources([]);
  setIsFileUploaded(false);
};
```

## Best Practices Applied

1. **Component Structure**:
   - Separated concerns with dedicated components
   - Used proper prop types and default props

2. **Error Handling**:
   - Implemented try-catch blocks for API calls
   - Added user-friendly error messages with toast notifications

3. **Code Organization**:
   - Grouped related functionality
   - Used descriptive variable and function names
   - Added comments for complex logic

## Future Improvements

1. Add unit and integration tests
2. Implement loading states for better UX
3. Add more file type support
4. Implement user authentication
5. Add document management features

## Resources Used
- [React Documentation](https://reactjs.org/)
- [Material-UI Documentation](https://mui.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)

## Version History
- **2024-09-02**: Initial learning log created
- **2024-09-01**: Implemented conditional UI rendering for file upload flow
