import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Send, Upload, FileText, Bot, User, Sparkles, X, Menu } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// Dynamic API URL — reads from env at build time, falls back to localhost
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function App() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Hello! I'm your RAG Assistant. Upload a PDF to get started." }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  // Auto-scroll to bottom
  const messagesEndRef = useRef(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(scrollToBottom, [messages]);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      await axios.post(`${API_URL}/upload-pdf/`, formData);
      setMessages(prev => [...prev, { role: "bot", text: "✅ PDF indexed successfully! Ask me anything about it." }]);
    } catch (error) {
      console.error("Upload failed", error);
      alert("Upload failed. Check backend console.");
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
      // Send chat history (excluding the first welcome string)
      const historyMsg = messages.slice(1).map(msg => ({ role: msg.role, text: msg.text }));
      
      const response = await axios.post(`${API_URL}/chat/`, { 
        question: input,
        history: historyMsg
      });
      const botMessage = { role: "bot", text: response.data.answer };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      setMessages((prev) => [...prev, { role: "bot", text: "❌ Error: Could not reach the brain." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden relative selection:bg-cyan-500/30">
      
      {/* Background Gradients (The Aurora Effect) */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
         <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-600/20 rounded-full mix-blend-screen filter blur-3xl opacity-30 animate-blob"></div>
         <div className="absolute top-0 right-1/4 w-96 h-96 bg-cyan-600/20 rounded-full mix-blend-screen filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>
         <div className="absolute -bottom-32 left-1/3 w-96 h-96 bg-pink-600/20 rounded-full mix-blend-screen filter blur-3xl opacity-30 animate-blob animation-delay-4000"></div>
      </div>

      {/* --- SIDEBAR --- */}
      <AnimatePresence mode="wait">
        {sidebarOpen && (
          <motion.div 
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -300, opacity: 0 }}
            className="w-80 bg-slate-900/50 backdrop-blur-xl border-r border-white/10 flex flex-col z-20 shadow-2xl absolute md:relative h-full"
          >
            <div className="p-6 border-b border-white/10 flex justify-between items-center">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-tr from-cyan-500 to-blue-600 rounded-lg shadow-lg shadow-cyan-500/20">
                  <Sparkles size={20} className="text-white" />
                </div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                  DocuChat
                </h1>
              </div>
              <button onClick={() => setSidebarOpen(false)} className="md:hidden text-slate-400 hover:text-white">
                <X size={20} />
              </button>
            </div>

            <div className="p-6 flex-1 flex flex-col gap-6">
              <div className="bg-slate-800/50 rounded-xl p-4 border border-white/5 hover:border-white/10 transition-all group">
                <label className="flex flex-col gap-3 cursor-pointer">
                  <span className="text-sm font-medium text-slate-300 group-hover:text-cyan-400 transition-colors flex items-center gap-2">
                    <Upload size={16} /> Upload PDF
                  </span>
                  <input 
                    type="file" 
                    accept=".pdf"
                    onChange={(e) => setFile(e.target.files[0])}
                    className="text-xs text-slate-500 file:mr-3 file:py-2 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-slate-700 file:text-slate-300 hover:file:bg-slate-600 transition-all"
                  />
                </label>
                {file && (
                  <div className="mt-3 flex items-center gap-2 text-xs text-emerald-400 bg-emerald-500/10 p-2 rounded-lg">
                    <FileText size={14} />
                    <span className="truncate">{file.name}</span>
                  </div>
                )}
                <button 
                  onClick={handleUpload}
                  disabled={uploading || !file}
                  className="mt-4 w-full py-2.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 rounded-lg font-medium text-sm transition-all shadow-lg shadow-blue-900/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {uploading ? (
                    <> <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"/> Indexing... </>
                  ) : (
                    "Process Document"
                  )}
                </button>
              </div>
            </div>

            <div className="p-4 border-t border-white/10 text-xs text-slate-500 text-center">
              Powered by Llama-3 & Pinecone
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* --- MAIN CHAT AREA --- */}
      <div className="flex-1 flex flex-col relative z-10">
        
        {/* Header (Mobile Toggle) */}
        {!sidebarOpen && (
          <div className="absolute top-4 left-4 z-50">
            <button onClick={() => setSidebarOpen(true)} className="p-2 bg-slate-800/80 backdrop-blur-md rounded-lg text-slate-300 hover:text-white border border-white/10">
              <Menu size={20} />
            </button>
          </div>
        )}

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 scroll-smooth">
          {messages.map((msg, index) => (
            <motion.div 
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div className={`max-w-[85%] md:max-w-[70%] rounded-2xl p-4 shadow-xl flex gap-4 backdrop-blur-sm border ${
                msg.role === "user" 
                  ? "bg-gradient-to-br from-blue-600 to-violet-600 text-white border-transparent rounded-tr-none" 
                  : "bg-slate-800/80 text-slate-100 border-white/10 rounded-tl-none"
              }`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                  msg.role === "user" ? "bg-white/20" : "bg-cyan-900/30 text-cyan-400"
                }`}>
                  {msg.role === "bot" ? <Bot size={18} /> : <User size={18} />}
                </div>
                <div className="leading-relaxed text-sm md:text-base whitespace-pre-wrap">
                  {msg.text}
                </div>
              </div>
            </motion.div>
          ))}
          
          {loading && (
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              className="flex justify-start"
            >
               <div className="bg-slate-800/80 backdrop-blur-sm p-4 rounded-2xl rounded-tl-none border border-white/10 flex items-center gap-3">
                 <Bot size={18} className="text-cyan-400 animate-pulse" />
                 <div className="flex gap-1">
                   <span className="w-2 h-2 bg-slate-500 rounded-full animate-bounce"></span>
                   <span className="w-2 h-2 bg-slate-500 rounded-full animate-bounce delay-100"></span>
                   <span className="w-2 h-2 bg-slate-500 rounded-full animate-bounce delay-200"></span>
                 </div>
               </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 md:p-6">
          <div className="max-w-4xl mx-auto relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-xl opacity-20 group-hover:opacity-40 blur transition duration-500"></div>
            <div className="relative flex items-center bg-slate-900/90 backdrop-blur-xl border border-white/10 rounded-xl p-2 shadow-2xl">
              <input 
                type="text" 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Ask something about your document..."
                className="flex-1 bg-transparent text-slate-100 placeholder-slate-500 px-4 py-3 focus:outline-none text-sm md:text-base"
              />
              <button 
                onClick={sendMessage}
                disabled={loading || !input.trim()}
                className="p-3 bg-slate-800 hover:bg-slate-700 text-cyan-400 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send size={20} />
              </button>
            </div>
          </div>
          <p className="text-center text-slate-600 text-xs mt-3">
            AI can make mistakes. Please verify important information.
          </p>
        </div>

      </div>
    </div>
  );
}