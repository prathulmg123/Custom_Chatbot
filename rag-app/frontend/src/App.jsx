import { useState, useRef, useEffect } from 'react';
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
  Divider,
  Backdrop,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Chip
} from '@mui/material';
import { 
  Upload as UploadIcon, 
  Send as SendIcon, 
  Delete as DeleteIcon,
  Person as PersonIcon,
  SmartToy as BotIcon
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
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const messagesEndRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    } else {
      setFile(null);
    }
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
      
      // Add the uploaded file to the list
      setUploadedFiles(prevFiles => [...prevFiles, {
        name: file.name,
        size: file.size,
        uploadedAt: new Date().toISOString()
      }]);
      
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

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

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

    // Add user's question to chat history
    const userMessage = { 
      role: 'user', 
      content: question,
      timestamp: new Date().toISOString()
    };
    
    setChatHistory(prev => [...prev, userMessage]);
    setQuestion('');
    
    try {
      setIsLoading(true);
      
      const response = await axios.post(`${API_URL}/query`, {
        query: question,
        top_k: 3
      });
      
      // Add assistant's response to chat history
      const assistantMessage = { 
        role: 'assistant', 
        content: response.data.answer,
        sources: response.data.sources,
        timestamp: new Date().toISOString()
      };
      
      setChatHistory(prev => [...prev, assistantMessage]);
      setAnswer(response.data.answer);
      setSources(response.data.sources);
    } catch (error) {
      console.error('Error asking question:', error);
      const errorMessage = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error while processing your request.',
        isError: true,
        timestamp: new Date().toISOString()
      };
      setChatHistory(prev => [...prev, errorMessage]);
      
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
    setFile(null);
    setUploadedFiles([]);
    setChatHistory([]);
    document.getElementById('file-upload').value = '';
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
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', py: 4, position: 'relative' }}>
      <Backdrop
        sx={{ 
          color: '#fff',
          zIndex: (theme) => theme.zIndex.drawer + 1,
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 2
        }}
        open={isLoading || isProcessing}
      >
        <CircularProgress color="inherit" />
        <Typography variant="h6" color="white">
          {isProcessing ? 'Processing your document...' : 'Generating answer...'}
        </Typography>
      </Backdrop>
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

          <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
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
                  {file && (
                    <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                      {`${(file.size / 1024).toFixed(2)} KB`}
                    </Typography>
                  )}
                  <Button
                    variant="contained"
                    onClick={uploadFile}
                    disabled={!file || isProcessing}
                    startIcon={isProcessing ? <CircularProgress size={20} /> : null}
                    sx={{
                      '&:disabled': {
                        backgroundColor: 'action.disabledBackground',
                        color: 'text.disabled'
                      }
                    }}
                  >
                    {isProcessing ? 'Processing...' : 'Upload'}
                  </Button>
                </Box>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Uploaded Files List */}
              {uploadedFiles.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Uploaded Files:
                  </Typography>
                  <Box sx={{ maxHeight: '150px', overflowY: 'auto', border: '1px solid', borderColor: 'divider', borderRadius: 1, p: 1 }}>
                    {uploadedFiles.map((file, index) => (
                      <Box 
                        key={index} 
                        sx={{ 
                          display: 'flex', 
                          justifyContent: 'space-between', 
                          alignItems: 'center',
                          p: 1,
                          bgcolor: 'background.paper',
                          borderRadius: 1,
                          mb: 1
                        }}
                      >
                        <Typography variant="body2" noWrap sx={{ maxWidth: '70%' }}>
                          {file.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {(file.size / 1024).toFixed(2)} KB
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                </Box>
              )}

              <Divider sx={{ my: 2 }} />
            </Box>
          </Paper>

          {/* Chat History */}
          <Paper 
            elevation={3} 
            sx={{ 
              p: 0,
              height: '70vh',
              display: 'flex',
              flexDirection: 'column',
              bgcolor: 'background.paper',
              borderRadius: 2,
              overflow: 'hidden',
              position: 'relative'
            }}
          >
            {/* Chat messages container */}
            <Box sx={{ 
              flex: 1, 
              overflowY: 'auto',
              p: 2,
              display: 'flex',
              flexDirection: 'column'
            }}>
              {chatHistory.length === 0 ? (
              <Box sx={{ 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center', 
                justifyContent: 'center', 
                height: '100%',
                textAlign: 'center',
                color: 'text.secondary',
                p: 3
              }}>
                <BotIcon sx={{ fontSize: 60, mb: 2, color: 'primary.main' }} />
                <Typography variant="h6" gutterBottom>
                  How can I help you today?
                </Typography>
                <Typography variant="body2">
                  Ask any question about your uploaded documents or start a conversation.
                </Typography>
              </Box>
            ) : (
              <List sx={{ width: '100%', bgcolor: 'background.paper' }}>
                {chatHistory.map((message, index) => (
                  <ListItem 
                    key={index} 
                    alignItems="flex-start"
                    sx={{
                      flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                      alignItems: 'flex-start',
                      mb: 2
                    }}
                  >
                    <ListItemAvatar>
                      {message.role === 'user' ? (
                        <Avatar sx={{ bgcolor: 'primary.main' }}>
                          <PersonIcon />
                        </Avatar>
                      ) : (
                        <Avatar sx={{ bgcolor: 'secondary.main' }}>
                          <BotIcon />
                        </Avatar>
                      )}
                    </ListItemAvatar>
                    <Box 
                      sx={{
                        maxWidth: '80%',
                        ml: message.role === 'user' ? 0 : 1,
                        mr: message.role === 'user' ? 1 : 0,
                        bgcolor: message.role === 'user' ? 'primary.light' : 'grey.100',
                        p: 2,
                        borderRadius: 2,
                        position: 'relative',
                        color: 'primary.contrastText',
                        borderTopLeftRadius: message.role === 'user' ? 12 : 4,
                        borderTopRightRadius: message.role === 'user' ? 4 : 12,
                      }}
                    >
                      <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }}>
                        {message.content}
                      </Typography>
                      {/* {message.sources && message.sources.length > 0 && (
                        <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {message.sources.map((source, idx) => (
                            <Chip 
                              key={idx} 
                              label={`Source ${idx + 1}`} 
                              size="small" 
                              color="primary"
                              variant="outlined"
                              sx={{ 
                                height: 20, 
                                fontSize: '0.65rem',
                                bgcolor: message.role === 'user' ? 'primary.dark' : 'background.paper'
                              }}
                            />
                          ))}
                        </Box>
                      )} */}
                      <Typography 
                        variant="caption" 
                        sx={{
                          display: 'block',
                          textAlign: 'right',
                          mt: 0.5,
                          color: message.role === 'user' ? 'rgba(255, 255, 255, 0.7)' : 'text.secondary',
                          fontSize: '0.6rem'
                        }}
                      >
                        {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </Typography>
                    </Box>
                  </ListItem>
                ))}
                {isLoading && (
                  <ListItem>
                    <ListItemAvatar>
                      <Avatar sx={{ bgcolor: 'secondary.main' }}>
                        <BotIcon />
                      </Avatar>
                    </ListItemAvatar>
                    <Box sx={{ 
                      bgcolor: 'grey.100', 
                      p: 2, 
                      borderRadius: 2,
                      borderTopLeftRadius: 4,
                      borderTopRightRadius: 12,
                    }}>
                      <CircularProgress size={20} />
                    </Box>
                  </ListItem>
                )}
                <div ref={messagesEndRef} />
              </List>
              )}
            </Box>
            
            {/* Input area fixed at bottom */}
            <Box sx={{ 
              p: 2, 
              borderTop: '1px solid',
              borderColor: 'divider',
              bgcolor: 'background.paper',
              position: 'sticky',
              bottom: 0,
              zIndex: 10
            }}>
              <Divider sx={{ my: 2 }} />
              <Box sx={{ 
                maxWidth: '800px', 
                mx: 'auto',
                display: 'flex',
                gap: 1
              }}>
                <TextField
                  fullWidth
                  variant="outlined"
                  size="small"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Type your question here..."
                  onKeyPress={(e) => e.key === 'Enter' && askQuestion()}
                  disabled={isLoading || !isFileUploaded}
                  sx={{ 
                    '& .MuiOutlinedInput-root': { 
                      borderRadius: 4,
                      bgcolor: 'background.paper',
                      pr: 1
                    } 
                  }}
                />
                <IconButton
                  color="primary"
                  onClick={askQuestion}
                  disabled={!question.trim() || isLoading || !isFileUploaded}
                  aria-label="Ask question"
                  sx={{ 
                    bgcolor: 'primary.main',
                    color: 'primary.contrastText',
                    '&:hover': {
                      bgcolor: 'primary.dark',
                    },
                    '&:disabled': {
                      bgcolor: 'action.disabledBackground',
                      color: 'action.disabled'
                    }
                  }}
                >
                  <SendIcon />
                </IconButton>
                <IconButton
                  onClick={clearAll}
                  color="inherit"
                  aria-label="Clear"
                  sx={{ ml: 1 }}
                >
                  <DeleteIcon />
                </IconButton>
              </Box>
              {!isFileUploaded && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', textAlign: 'center' }}>
                  Please upload a document to start chatting
                </Typography>
              )}
            </Box>
          </Paper>
        </Box>
      </Container>
    </Box>
  );
}

export default App;
