import { useState } from 'react';
import { toast } from 'react-toastify';
import axios from 'axios';
import 'react-toastify/dist/ReactToastify.css';

// Material-UI Components
import { 
  Box, 
  Button, 
  Container, 
  Paper, 
  TextField, 
  Typography, 
  CircularProgress,
  IconButton,
  Divider
} from '@mui/material';
import { 
  Upload as UploadIcon, 
  Send as SendIcon, 
  Delete as DeleteIcon 
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const API_URL = '/api';

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isFileUploaded, setIsFileUploaded] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const uploadFile = async () => {
    if (!file) {
      toast.error('Please select a file first');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setIsProcessing(true);
      const response = await axios.post(`${API_URL}/ingest`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      toast.success(`Successfully file uploaded`, {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
      
      setFile(null);
      document.getElementById('file-upload').value = '';
      setIsFileUploaded(true);
    } catch (error) {
      console.error('Error uploading file:', error);
      toast.error(error.response?.data?.detail || 'Failed to process document', {
        position: "top-right",
        autoClose: 5000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) {
      toast.error('Please enter a question', {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
      return;
    }

    try {
      setIsLoading(true);
      setAnswer('');
      setSources([]);
      
      const response = await axios.post(`${API_URL}/query`, {
        query: question,
        top_k: 3
      });
      
      setAnswer(response.data.answer);
      setSources(response.data.sources);
    } catch (error) {
      console.error('Error asking question:', error);
      toast.error(error.response?.data?.detail || 'Failed to get answer', {
        position: "top-right",
        autoClose: 5000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const clearAll = () => {
    setQuestion('');
    setAnswer('');
    setSources([]);
    setIsFileUploaded(false);
  };

  // Styled components
  const VisuallyHiddenInput = styled('input')({
    clip: 'rect(0 0 0 0)',
    clipPath: 'inset(50%)',
    height: 1,
    overflow: 'hidden',
    position: 'absolute',
    bottom: 0,
    left: 0,
    whiteSpace: 'nowrap',
    width: 1,
  });

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', py: 4 }}>
      <Container maxWidth="md">
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h3" component="h1" gutterBottom color="text.primary">
              Document Q&A with RAG
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Upload documents and ask questions about their content
            </Typography>
          </Box>

          <Paper elevation={3} sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Upload a document (PDF or text)
                </Typography>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                  <Button
                    component="label"
                    variant="outlined"
                    startIcon={<UploadIcon />}
                    sx={{ flex: 1, justifyContent: 'flex-start' }}
                  >
                    {file ? file.name : 'Choose File'}
                    <VisuallyHiddenInput 
                      id="file-upload"
                      type="file"
                      accept=".pdf,.txt"
                      onChange={handleFileChange}
                    />
                  </Button>
                  <Button
                    variant="contained"
                    onClick={uploadFile}
                    disabled={!file || isProcessing}
                    startIcon={isProcessing ? <CircularProgress size={20} /> : null}
                  >
                    {isProcessing ? 'Processing...' : 'Upload'}
                  </Button>
                </Box>
              </Box>

              <Divider />

              {isFileUploaded && (
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Ask a question about the document
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    fullWidth
                    variant="outlined"
                    size="small"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Type your question here..."
                    onKeyPress={(e) => e.key === 'Enter' && askQuestion()}
                    disabled={isLoading}
                    sx={{ flex: 1 }}
                  />
                  <IconButton
                    color="primary"
                    onClick={askQuestion}
                    disabled={!question.trim() || isLoading}
                    aria-label="Ask question"
                  >
                    <SendIcon />
                  </IconButton>
                  <IconButton
                    onClick={clearAll}
                    color="inherit"
                    aria-label="Clear"
                  >
                    <DeleteIcon />
                  </IconButton>
                </Box>
              </Box>
              )}
            </Box>
          </Paper>

          {(answer || sources.length > 0) && (
            <Paper elevation={3} sx={{ p: 3 }}>
              {isLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                  <CircularProgress />
                </Box>
              ) : (
                <>
                  {answer && (
                    <Box sx={{ mb: 4 }}>
                      <Typography variant="h6" gutterBottom>
                        Answer:
                      </Typography>
                      <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }}>
                        {answer}
                      </Typography>
                    </Box>
                  )}

                  {sources.length > 0 && (
                    <Box>
                      <Typography variant="h6" gutterBottom>
                        Sources:
                      </Typography>
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        {sources.map((source, index) => (
                          <Paper 
                            key={index} 
                            variant="outlined" 
                            sx={{ p: 2, bgcolor: 'background.paper' }}
                          >
                            <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                              Source {index + 1}:
                            </Typography>
                            <Typography variant="body2">{source.text}</Typography>
                            {source.metadata && (
                              <Typography variant="caption" color="primary" display="block" sx={{ mt: 1 }}>
                                Page: {source.metadata.page || 'N/A'}
                              </Typography>
                            )}
                          </Paper>
                        ))}
                      </Box>
                    </Box>
                  )}
                </>
              )}
            </Paper>
          )}
        </Box>
      </Container>
    </Box>
  );
}

export default App;
