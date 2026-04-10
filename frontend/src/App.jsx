import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { 
  Send, 
  FileText, 
  User, 
  Menu, 
  Plus, 
  Command, 
  MoreHorizontal,
  Paperclip,
  CheckCircle2,
  AlertCircle,
  Loader2
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from 'react-markdown';

/**
 * Safer API URL definition to avoid 'import.meta' build issues in certain environments.
 * In a local Vite project, you would use import.meta.env.VITE_API_URL.
 */
const getApiUrl = () => {
  try {
    // Attempt to access Vite env, fallback if it fails or isn't defined
    return (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) 
      || "http://127.0.0.1:8000";
  } catch (e) {
    return "http://127.0.0.1:8000";
  }
};

const API_URL = getApiUrl();

export default function App() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Hello. I am connected to your knowledge base. Upload a document to begin analysis." }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  const [sessionId, setSessionId] = useState(() => Math.random().toString(36).substring(2, 10));

  const handleNewChat = () => {
    setSessionId(Math.random().toString(36).substring(2, 10));
    setFile(null);
    setMessages([
      { role: "bot", text: "Session cleared. Ready for a new document." }
    ]);
  };
  
  const messagesEndRef = useRef(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(scrollToBottom, [messages]);

  const handleUpload = async (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setUploading(true);
    
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("session_id", sessionId);

    try {
      await axios.post(`${API_URL}/upload-pdf/`, formData);
      
      setMessages(prev => [...prev, { 
        role: "bot", 
        text: `Document "${selectedFile.name}" has been successfully indexed. I've processed the context into Pinecone. How can I help you analyze it?` 
      }]);
    } catch (error) {
      console.error("Upload failed", error);
      setMessages(prev => [...prev, { role: "bot", text: "Error: Failed to process the document. Please check your backend connection." }]);
    } finally {
      setUploading(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMessage = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const historyMsg = messages.slice(1).map(msg => ({ role: msg.role, text: msg.text }));
      
      const response = await axios.post(`${API_URL}/chat/`, { 
        question: input,
        history: historyMsg,
        session_id: sessionId
      });
      const botMessage = { role: "bot", text: response.data.answer };
      
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      setMessages((prev) => [...prev, { role: "bot", text: "Error: Could not reach the brain. Ensure your Render backend is active." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#09090b] text-zinc-300 font-sans overflow-hidden selection:bg-zinc-800 selection:text-white">
      
      {/* --- SIDEBAR --- */}
      <AnimatePresence mode="wait">
        {sidebarOpen && (
          <motion.div 
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -300, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="w-72 bg-[#09090b] border-r border-zinc-800/50 flex flex-col z-30 absolute md:relative h-full"
          >
            {/* Logo Area */}
            <div className="h-16 flex items-center px-6 border-b border-zinc-800/50 justify-between">
              <div className="flex items-center gap-3 text-zinc-100">
                <div className="w-7 h-7 bg-white text-black rounded-md flex items-center justify-center">
                  <Command size={16} strokeWidth={2.5} />
                </div>
                <span className="font-semibold tracking-tight text-sm uppercase">DocuChat</span>
              </div>
              <button 
                onClick={() => setSidebarOpen(false)} 
                className="md:hidden text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                <MoreHorizontal size={20} />
              </button>
            </div>

            {/* Actions Area */}
            <div className="p-4 flex-1 flex flex-col gap-4">
              <button 
                onClick={handleNewChat}
                className="w-full flex items-center gap-2 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-100 px-4 py-2.5 rounded-lg text-sm font-medium transition-all"
              >
                <Plus size={16} className="text-zinc-400" />
                New Thread
              </button>

              <div className="mt-4">
                <h3 className="text-xs font-semibold text-zinc-500 tracking-wider uppercase mb-3 px-2">Knowledge Base</h3>
                
                <div className="relative group">
                  <input 
                    type="file" 
                    accept=".pdf"
                    onChange={(e) => handleUpload(e.target.files[0])}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    disabled={uploading}
                  />
                  <div className={`border border-dashed rounded-xl p-5 flex flex-col items-center justify-center text-center transition-colors ${
                      file 
                        ? "border-zinc-700 bg-zinc-900/50" 
                        : "border-zinc-800 bg-zinc-900/20 group-hover:bg-zinc-900/60 group-hover:border-zinc-700"
                    }`}
                  >
                    {uploading ? (
                      <Loader2 size={24} className="text-zinc-400 animate-spin mb-2" />
                    ) : file ? (
                      <CheckCircle2 size={24} className="text-zinc-300 mb-2" strokeWidth={1.5} />
                    ) : (
                      <FileText size={24} className="text-zinc-500 mb-2" strokeWidth={1.5} />
                    )}
                    
                    <span className="text-sm font-medium text-zinc-300">
                      {uploading ? "Indexing..." : file ? file.name : "Upload PDF"}
                    </span>
                    <span className="text-xs text-zinc-600 mt-1">
                      {uploading ? "Creating Vectors" : file ? "Ready to query" : "PDF format only"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Footer Status */}
            <div className="p-4 border-t border-zinc-800/50">
              <div className="flex items-center gap-2 text-xs text-zinc-500 bg-zinc-900/50 py-2 px-3 rounded-md">
                <div className="w-2 h-2 rounded-full bg-emerald-500/80 animate-pulse" />
                System Active
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* --- MAIN CHAT AREA --- */}
      <div className="flex-1 flex flex-col relative w-full h-full">
        
        {/* Navigation / Header */}
        <header className="h-16 flex items-center px-4 border-b border-zinc-800/50 bg-[#09090b]/80 backdrop-blur-md sticky top-0 z-20">
          {!sidebarOpen && (
            <button 
              onClick={() => setSidebarOpen(true)} 
              className="mr-4 text-zinc-400 hover:text-zinc-100 transition-colors"
            >
              <Menu size={20} />
            </button>
          )}
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-zinc-500">Instance</span>
            <span className="text-xs font-mono bg-zinc-800/50 text-zinc-400 px-2 py-1 rounded">
              {sessionId}
            </span>
          </div>
        </header>

        {/* Chat Feed */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 scroll-smooth">
          <div className="max-w-3xl mx-auto space-y-10 pb-32">
            {messages.map((msg, index) => (
              <motion.div 
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
                className={`flex gap-5 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
              >
                {/* Visual Identifier (Avatar) */}
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border ${
                  msg.role === "user" 
                    ? "bg-zinc-800 border-zinc-700 text-zinc-400" 
                    : "bg-white text-black border-zinc-200 shadow-sm"
                }`}>
                  {msg.role === "bot" ? <Command size={14} strokeWidth={2.5} /> : <User size={14} />}
                </div>

                {/* Message Body */}
                <div className={`flex flex-col max-w-[85%] ${msg.role === "user" ? "items-end" : "items-start"}`}>
                  <span className="text-[10px] font-bold text-zinc-600 mb-1.5 uppercase tracking-widest">
                    {msg.role === "bot" ? "Assistant" : "Query"}
                  </span>
                  <div className={`text-[15px] leading-relaxed tracking-tight ${
                    msg.role === "user" 
                      ? "bg-zinc-900 text-zinc-100 px-5 py-3 rounded-2xl rounded-tr-sm border border-zinc-800" 
                      : "text-zinc-200 px-0 py-0"
                  }`}>
                    {msg.role === "user" ? (
                      msg.text
                    ) : (
                      <ReactMarkdown 
                        className="prose prose-invert max-w-none text-[15px] leading-relaxed"
                        components={{
                          p: ({node, ...props}) => <p className="mb-3 last:mb-0" {...props} />,
                          ul: ({node, ...props}) => <ul className="list-disc pl-5 space-y-1 my-3 marker:text-zinc-600" {...props} />,
                          ol: ({node, ...props}) => <ol className="list-decimal pl-5 space-y-1 my-3 marker:text-zinc-600" {...props} />,
                          li: ({node, ...props}) => <li className="pl-1" {...props} />,
                          strong: ({node, ...props}) => <strong className="font-semibold text-zinc-100" {...props} />,
                          a: ({node, ...props}) => <a className="text-zinc-300 hover:text-white underline underline-offset-4 decoration-zinc-700 hover:decoration-zinc-400 transition-colors" {...props} />,
                          code: ({node, inline, ...props}) => inline 
                            ? <code className="bg-zinc-800/50 text-zinc-300 px-1.5 py-0.5 rounded-md text-[13px] font-mono border border-zinc-700/50" {...props} />
                            : <code className="block bg-[#050506] text-zinc-300 p-4 rounded-xl my-4 text-[13px] font-mono overflow-x-auto border border-zinc-800/80 shadow-inner" {...props} />
                        }}
                      >
                        {msg.text}
                      </ReactMarkdown>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
            
            {loading && (
              <motion.div 
                initial={{ opacity: 0 }} 
                animate={{ opacity: 1 }} 
                className="flex gap-5 flex-row max-w-3xl mx-auto"
              >
                <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border bg-white text-black border-zinc-200">
                  <Command size={14} strokeWidth={2.5} />
                </div>
                <div className="flex flex-col items-start">
                  <span className="text-[10px] font-bold text-zinc-600 mb-1.5 uppercase tracking-widest">Assistant</span>
                  <div className="flex gap-1.5 py-2">
                    <span className="w-1 h-1 bg-zinc-600 rounded-full animate-bounce"></span>
                    <span className="w-1 h-1 bg-zinc-600 rounded-full animate-bounce delay-75"></span>
                    <span className="w-1 h-1 bg-zinc-600 rounded-full animate-bounce delay-150"></span>
                  </div>
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} className="h-2" />
          </div>
        </div>

        {/* Input Bar Section */}
        <div className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-[#09090b] via-[#09090b] to-transparent pt-20 pb-8 px-4 md:px-8 pointer-events-none">
          <div className="max-w-3xl mx-auto pointer-events-auto">
            <div className="relative flex items-end gap-2 bg-[#121214] border border-zinc-800 focus-within:border-zinc-600 rounded-2xl p-2 transition-all shadow-2xl shadow-black/40">
              
              <button className="p-3 text-zinc-500 hover:text-zinc-300 transition-colors shrink-0">
                <Paperclip size={18} />
              </button>

              <textarea 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                rows={1}
                placeholder="Message your document..."
                className="w-full max-h-48 bg-transparent text-zinc-100 placeholder-zinc-600 px-2 py-3 focus:outline-none resize-none text-[15px]"
                style={{ minHeight: '46px' }}
              />

              <button 
                onClick={sendMessage}
                disabled={loading || !input.trim()}
                className="p-2.5 mb-0.5 mr-0.5 bg-white hover:bg-zinc-200 text-black rounded-xl transition-colors disabled:opacity-0 disabled:scale-95 shrink-0 flex items-center justify-center shadow-lg active:scale-95"
              >
                <Send size={16} strokeWidth={2.5} />
              </button>
            </div>
            
            <div className="flex items-center justify-center gap-1.5 mt-4 text-[10px] text-zinc-600 font-medium uppercase tracking-tighter">
              <AlertCircle size={10} />
              <span>Contextual results from Llama-3-8B & Pinecone Serverless</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}