import React, { useEffect, useRef, useState } from 'react';
import { 
  FileText, 
  Upload, 
  Settings, 
  Mic, 
  Send, 
  X, 
  CheckCircle, 
  Clock, 
  AlertTriangle,
  ExternalLink,
  User,
  Bot
} from 'lucide-react';
import CompareToggle from './CompareToggle.jsx';

const App = () => {
  const [compareMode, setCompareMode] = useState(false);
  const [voiceMode, setVoiceMode] = useState(false);
  const recognitionRef = useRef(null);
  useEffect(() => {
    const WSR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (WSR) {
      const rec = new WSR();
      rec.lang = 'en-US';
      rec.continuous = false;
      rec.interimResults = false;
      rec.onresult = (e) => {
        const transcript = e.results[0][0].transcript;
        setMessage((prev) => (prev ? prev + ' ' : '') + transcript);
      };
      rec.onerror = (e) => console.warn('Speech error', e);
      recognitionRef.current = rec;
    }
  }, []);
  const [showSettings, setShowSettings] = useState(false);
  const [message, setMessage] = useState('');
  const [groqApiKey, setGroqApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState('llama-3.1-70b-versatile');
  const [dragOver, setDragOver] = useState(false);
  
  const [uploadedFiles, setUploadedFiles] = useState([
    {
      name: 'Medical_Guidelines_2024.pdf',
      size: '2.4 MB',
      pages: 45,
      status: 'completed'
    },
    {
      name: 'Product_Catalog_Q1.pdf',
      size: '1.8 MB',
      pages: 23,
      status: 'processing',
      progress: 67
    }
  ]);

  const [messages, setMessages] = useState([
    {
      id: '1',
      type: 'bot',
      content: "Hello! I'm your medicine support assistant. I can help you find information from your uploaded documents. What would you like to know?",
      timestamp: '10:52:56 PM'
    },
    {
      id: '2',
      type: 'user',
      content: 'What are the side effects of the medication mentioned in the guidelines?',
      timestamp: '10:52:56 PM'
    },
    {
      id: '3',
      type: 'bot',
      content: "Based on the Medical Guidelines document, the common side effects include nausea, dizziness, and fatigue. However, I must emphasize that this information is for reference only and should not replace professional medical advice.",
      timestamp: '10:52:56 PM',
      references: [
        { page: 12, percentage: 94 },
        { page: 15, percentage: 87 }
      ],
      escalateOption: true
    }
  ]);

  const models = [
    'llama-3.1-70b-versatile',
    'llama-3.1-8b-instant',
    'mixtral-8x7b-32768',
    'gemma-7b-it'
  ];

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files || []);
    for (const f of files) {
      try {
        setUploadedFiles((prev) => [...prev, { name: f.name, size: `${(f.size/1e6).toFixed(1)} MB`, pages: '-', status: 'processing', progress: 10 }]);
        const { uploadDocument } = await import('./api');
        await uploadDocument(f);
        setUploadedFiles((prev) => prev.map((item) => item.name === f.name ? { ...item, status: 'completed', progress: 100 } : item));
      } catch (err) {
        console.error('Upload failed', err);
        setUploadedFiles((prev) => prev.map((item) => item.name === f.name ? { ...item, status: 'error' } : item));
      }
    }
  };

  const handleSendMessage = async () => {
    if (!message.trim()) return;
    const newMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: message,
      timestamp: new Date().toLocaleTimeString()
    };
    setMessages((prev) => [...prev, newMessage]);
    const q = message;
    setMessage('');
    try {
      const { sendChat } = await import('./api');
      const res = await sendChat(q);
      const answer = res.answer || res.response || 'No response';
      setMessages((prev) => [...prev, {
        id: Date.now().toString() + '-bot',
        type: 'bot',
        content: answer,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } catch (e) {
      console.error(e);
      setMessages((prev) => [...prev, {
        id: Date.now().toString() + '-bot-err',
        type: 'bot',
        content: 'Error contacting backend. Check server logs.',
        timestamp: new Date().toLocaleTimeString()
      }]);
    }
  };

  const handleValidateApiKey = async () => {
    try {
      const { updateLLMSettings } = await import('./api');
      await updateLLMSettings({ apiKey: groqApiKey, model: selectedModel });
      alert('Settings saved');
    } catch (e) {
      console.error(e);
      alert('Failed to save settings');
    }
  };

  const handleModelSelect = async () => {
    try {
      const { updateLLMSettings } = await import('./api');
      await updateLLMSettings({ apiKey: groqApiKey, model: selectedModel });
    } catch (e) {
      console.error(e);
    } finally {
      setShowSettings(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-96 bg-white border-r border-gray-200 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <h1 className="font-semibold text-gray-900">RAG Support Bot</h1>
          </div>
          <button
            onClick={() => setShowSettings(true)}
            className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>

        {/* Documents Section */}
        <div className="p-4 flex-1 overflow-y-auto">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-medium text-gray-900">Documents</h2>
            <CompareToggle checked={compareMode} onChange={setCompareMode} />
          </div>

          {/* Upload Area */}
          <div
            className={`border-2 border-dashed rounded-lg p-6 text-center mb-4 transition-colors ${
              dragOver 
                ? 'border-blue-400 bg-blue-50' 
                : 'border-gray-300 hover:border-gray-400'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
            <h3 className="font-medium text-gray-900 mb-1">Upload PDF Documents</h3>
            <p className="text-sm text-gray-500 mb-2">
              Drag and drop or click to browse
            </p>
            <p className="text-xs text-gray-400">Supports PDF files up to 10MB</p>
          </div>

          {/* Uploaded Files */}
          <div className="space-y-2 mb-4">
            {uploadedFiles.map((file, index) => (
              <div key={index} className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-gray-500 flex-shrink-0" />
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium text-gray-900 truncate">{file.name}</p>
                      <p className="text-xs text-gray-500">{file.size} • {file.pages} pages</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {file.status === 'completed' && (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    )}
                    {file.status === 'processing' && (
                      <Clock className="w-4 h-4 text-yellow-500" />
                    )}
                    {file.status === 'error' && (
                      <AlertTriangle className="w-4 h-4 text-red-500" />
                    )}
                  </div>
                </div>
                {file.status === 'processing' && file.progress && (
                  <div className="mt-2">
                    <div className="text-xs text-gray-500 mb-1">Processing... {file.progress}%</div>
                    <div className="w-full bg-gray-200 rounded-full h-1">
                      <div 
                        className="bg-blue-600 h-1 rounded-full transition-all duration-300"
                        style={{ width: `${file.progress}%` }}
                      />
                    </div>
                  </div>
                )}
                {file.status === 'completed' && (
                  <div className="flex gap-2 mt-2">
                    <button className="text-xs px-2 py-1 bg-gray-200 rounded text-gray-600 hover:bg-gray-300 transition-colors">
                      👁️ Preview
                    </button>
                    <button className="text-xs px-2 py-1 bg-blue-100 text-blue-600 rounded hover:bg-blue-200 transition-colors">
                      Chat with PDF
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Medical Disclaimer */}
        <div className="bg-orange-50 border-b border-orange-200 px-4 py-2">
          <div className="flex items-center gap-2 text-sm text-orange-700">
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
            <span>Medical information provided is for reference only. Consult healthcare professionals for medical advice.</span>
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex gap-3 ${msg.type === 'user' ? 'justify-end' : ''}`}>
              {msg.type === 'bot' && (
                <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-white" />
                </div>
              )}
              <div className={`max-w-2xl ${msg.type === 'user' ? 'order-1' : ''}`}>
                <div className={`rounded-lg p-4 ${
                  msg.type === 'user' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-white border border-gray-200'
                }`}>
                  <p className="text-sm leading-relaxed">{msg.content}</p>
                  {msg.references && (
                    <div className="flex gap-2 mt-3">
                      {msg.references.map((ref, idx) => (
                        <button
                          key={idx}
                          className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs hover:bg-blue-200 transition-colors"
                        >
                          📄 Page {ref.page} ({ref.percentage}%)
                        </button>
                      ))}
                    </div>
                  )}
                  {msg.escalateOption && (
                    <button className="inline-flex items-center gap-1 mt-3 px-3 py-1 bg-red-100 text-red-700 rounded-md text-xs hover:bg-red-200 transition-colors">
                      <ExternalLink className="w-3 h-3" />
                      Escalate to Human
                    </button>
                  )}
                </div>
                <div className="text-xs text-gray-500 mt-1 text-right">
                  {msg.timestamp}
                </div>
              </div>
              {msg.type === 'user' && (
                <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                  <User className="w-4 h-4 text-white" />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 p-4">
          <div className="flex gap-3 items-end">
            <div className="flex-1 relative">
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Ask about your documents..."
                className="w-full px-4 py-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={3}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  const rec = recognitionRef.current;
                  if (!rec) return alert('Speech Recognition not supported in this browser.');
                  try { rec.start(); } catch {}
                  setVoiceMode(true);
                  rec.onend = () => setVoiceMode(false);
                }}
                className={`p-3 rounded-lg transition-colors ${
                  voiceMode
                    ? 'bg-blue-100 text-blue-600'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
                title="Voice Assistant"
              >
                <Mic className="w-5 h-5" />
              </button>
              <button
                onClick={handleSendMessage}
                disabled={!message.trim()}
                className="p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
          <div className="flex justify-center mt-3 text-xs text-gray-500 space-x-4">
            <span>Powered by RAG Technology</span>
            <span>•</span>
            <span>Enterprise Security</span>
            <span>•</span>
            <span>Multi-Modal AI</span>
          </div>
        </div>
      </div>

      {/* Settings Popup */}
      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">API Configuration</h2>
              <button
                onClick={() => setShowSettings(false)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <div className="space-y-6">
              {/* API Key */}
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">
                  Groq API key:
                </label>
                <div className="flex gap-2">
                  <input
                    type="password"
                    value={groqApiKey}
                    onChange={(e) => setGroqApiKey(e.target.value)}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Enter your Groq API key"
                  />
                  <button
                    onClick={handleValidateApiKey}
                    className="px-4 py-2 bg-gray-100 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-200 transition-colors"
                  >
                    VALIDATE
                  </button>
                </div>
              </div>

              {/* Models */}
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">
                  Models
                </label>
                <div className="flex gap-2">
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    {models.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={handleModelSelect}
                    className="px-6 py-2 bg-gray-100 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-200 transition-colors"
                  >
                    OK
                  </button>
                </div>
              </div>

              {/* Selected Model Display */}
              <div>
                <div className="px-4 py-3 bg-gray-50 border border-gray-300 rounded-md">
                  <span className="text-sm text-gray-700">{selectedModel}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;