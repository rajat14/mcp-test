
import React, { useState, useEffect, useRef } from 'react';
//import { Send, Bot, User, Settings, Wrench, ChevronDown, ChevronUp, Copy, Check, Server } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
// Add new imports at the top:
import { Send, Bot, User, Settings, Wrench, ChevronDown, ChevronUp, Copy, Check, Server, Activity, DollarSign, Hash } from 'lucide-react';
const MCPChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [tools, setTools] = useState([]);
  const [currentToolCall, setCurrentToolCall] = useState({ name: null, args: null });
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [expandedTools, setExpandedTools] = useState({});
  const [copiedStates, setCopiedStates] = useState({});
  const [sessionMetrics, setSessionMetrics] = useState(null);
  const[toolResults, setToolResults] = useState([]);
  
  // New state for server management
  const [availableServers, setAvailableServers] = useState([]);
  const [activeServer, setActiveServer] = useState('');
  const [isServerSwitching, setIsServerSwitching] = useState(false);
  const [serverManagerEnabled, setServerManagerEnabled] = useState(false); // ## ADDED ##
  
  const messagesEndRef = useRef(null);
  const API_URL = 'http://localhost:8008'; // Your FastAPI URL
  // EXL logo URL
  const LOGO_URL = 'https://i.pinimg.com/originals/dc/30/1d/dc301dd6fac108a2a60f103e01539f04.jpg';
  
  //  Data Harbor logo
  const DATAHARBOR_LOGO_URL = "https://dataharbor-dev.exlservice.com/dataharbor-logo.svg"; // Add your DataHarbor logo URL
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  useEffect(() => {
    fetchServers();
    fetchTools();
  }, []);
  const fetchServers = async () => {
  try {
    const response = await fetch(`${API_URL}/servers`);
    const data = await response.json();
    setAvailableServers(data.available_servers || []);
    setActiveServer(data.active_server || '');
    setServerManagerEnabled(data.server_manager_enabled || false); // ## ADDED ##
  } catch (error) {
    console.error('Error fetching servers:', error);
  }
};
  // const fetchServers = async () => {
  //   try {
  //     const response = await fetch(`${API_URL}/servers`, {
  //       headers: { 'Content-Type': 'application/json' }
  //     });
  //     const data = await response.json();
  //     setAvailableServers(data.available_servers || []);
  //     setActiveServer(data.active_server || '');
  //   } catch (error) {
  //     console.error('Error fetching servers:', error);
  //   }
  // };
  const fetchTools = async () => {
    try {
      const response = await fetch(`${API_URL}/tools`, {
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await response.json();
      setTools(data.tools || []);
    } catch (error) {
      console.error('Error fetching tools:', error);
    }
  };
  // Add function to fetch session metrics:
  const fetchSessionMetrics = async () => {
    try {
      const response = await fetch(`${API_URL}/session-metrics`, {
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await response.json();
      setSessionMetrics(data.metrics);
    } catch (error) {
      console.error('Error fetching session metrics:', error);
    }
  };
  // Add useEffect to periodically fetch metrics:
  useEffect(() => {
    fetchSessionMetrics();
    const interval = setInterval(fetchSessionMetrics, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);
  const handleServerSwitch = async (serverName) => {
    // ## ADDED ##: Prevent switching if manager is on
    if (serverManagerEnabled || serverName === activeServer || isServerSwitching) return;
    //if (serverName === activeServer || isServerSwitching) return;
    
    setIsServerSwitching(true);
    try {
      const response = await fetch(`${API_URL}/switch-server`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ server_name: serverName })
      });
      if (response.ok) {
        const data = await response.json();
        setActiveServer(data.active_server);
        // Refresh tools after switching server
        await fetchTools();
        
        // Add a system message to indicate server switch
        const switchMessage = {
          role: 'system',
          content: `Switched to server: ${serverName}`,
          id: Date.now(),
          type: 'system'
        };
        setMessages(prev => [...prev, switchMessage]);
      } else {
        throw new Error(`Failed to switch server: ${response.status}`);
      }
    } catch (error) {
      console.error('Error switching server:', error);
      const errorMessage = {
        role: 'assistant',
        content: `Error switching to server ${serverName}: ${error.message}`,
        id: Date.now(),
        type: 'error'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsServerSwitching(false);
    }
  };
  const copyToClipboard = async (text, id) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedStates(prev => ({ ...prev, [id]: true }));
      setTimeout(() => {
        setCopiedStates(prev => ({ ...prev, [id]: false }));
      }, 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };
  const toggleToolExpansion = (toolId) => {
    setExpandedTools(prev => ({
      ...prev,
      [toolId]: !prev[toolId]
    }));
  };
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };
  const processMessages = (newMessages) => {
    const processedMessages = [];
    let currentToolCall = { name: null, args: null };
    for (const message of newMessages) {
      if (message.role === 'user' && typeof message.content === 'string') {
        processedMessages.push({
          ...message,
          id: Date.now() + Math.random(),
          type: 'user_text'
        });
      }
      if (message.role === 'user' && Array.isArray(message.content)) {
        for (const content of message.content) {
          if (content.type === 'tool_result') {
            processedMessages.push({
              role: 'assistant',
              content: content,
              id: Date.now() + Math.random(),
              type: 'tool_result',
              toolCall: currentToolCall
            });
          }
        }
      }
      if (message.role === 'assistant' && typeof message.content === 'string') {
        processedMessages.push({
          ...message,
          id: Date.now() + Math.random(),
          type: 'assistant_text'
        });
      }
      if (message.role === 'assistant' && Array.isArray(message.content)) {
        for (const content of message.content) {
          if (content.type === 'tool_use') {
            currentToolCall = {
              name: content.name,
              args: content.input
            };
          }
        }
      }
    }
    return processedMessages;
  };
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;
    setIsLoading(true);
    const userMessage = {
      role: 'user',
      content: query,
      id: Date.now(),
      type: 'user_text'
    };
    // Add the user message immediately to show it in the UI
    setMessages(prev => [...prev, userMessage]);
    const currentQuery = query;
    setQuery(''); // Clear input immediately
    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: currentQuery })
      });
      if (response.ok) {
        const data = await response.json();
        const processedMessages = processMessages(data.messages);
        
        // Filter out the user message from API response to avoid duplication
        const assistantMessages = processedMessages.filter(msg => msg.role === 'assistant');
        
        // Append only the assistant messages to existing messages
        setMessages(prev => [...prev, ...assistantMessages]);
      } else {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
    } catch (error) {
      console.error('Error processing query:', error);
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${error.message}`,
        id: Date.now() + 1,
        type: 'error'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };
  const renderMessage = (message) => {
    const messageId = `msg-${message.id}`;
    if (message.type === 'system') {
      return (
        <div key={message.id} className="flex justify-center mb-4">
          <div className="bg-blue-100 border border-blue-200 text-blue-800 px-4 py-2 rounded-full text-sm">
            <Server size={14} className="inline mr-1" />
            {message.content}
          </div>
        </div>
      );
    }
    if (message.type === 'user_text') {
      return (
        <div key={message.id} className="flex justify-end mb-6">
          <div className="flex items-start max-w-3xl">
            <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-4 rounded-2xl rounded-br-md shadow-lg">
              <p className="whitespace-pre-wrap break-words">{message.content}</p>
            </div>
            <div className="ml-3 p-2 bg-blue-100 rounded-full">
              <User size={20} className="text-blue-600" />
            </div>
          </div>
        </div>
      );
    }
    if (message.type === 'assistant_text') {
      return (
        <div key={message.id} className="flex justify-start mb-6">
          <div className="flex items-start max-w-3xl">
            <div className="mr-3 p-2 bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-full">
              <Bot size={20} className="text-white" />
            </div>
            <div className="bg-white p-4 rounded-2xl rounded-bl-md shadow-lg border border-gray-100">
            <div
              className="text-gray-800 whitespace-pre-wrap break-words"
              dangerouslySetInnerHTML={{ __html: message.content }}/>
            </div>
          </div>
        </div>
      );
    }
    if (message.type === 'tool_result') {
      let parsedContent;
      try{
        parsedContent = JSON.parse(message.content.content[0].text);
      } catch (e) {
        parsedContent = {error: message.content.content[0].text};
      }
      const toolData = {
        name: message.toolCall.name,
        args: message.toolCall.args,
        content: parsedContent,
      };
      return (
        <div key={message.id} className="flex justify-start mb-6">
          <div className="flex items-start max-w-4xl w-full">
            <div className="mr-3 p-2 bg-gradient-to-r from-orange-400 to-pink-400 rounded-full">
              <Wrench size={20} className="text-white" />
            </div>
            <div className="bg-gradient-to-br from-gray-50 to-white p-4 rounded-2xl rounded-bl-md shadow-lg border border-gray-200 w-full">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-gray-800 flex items-center">
                  <Wrench size={16} className="mr-2 text-orange-500" />
                  Tool Result: {toolData.name}
                </h4>
                <button
                  onClick={() => toggleToolExpansion(messageId)}
                  className="p-1 hover:bg-gray-100 rounded transition-colors"
                >
                  {expandedTools[messageId] ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>
              </div>
              
              {expandedTools[messageId] && (
                <div className="space-y-3">
                  <div className="bg-blue-50 p-3 rounded-xl">
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="font-medium text-blue-800">Arguments</h5>
                      <button
                        onClick={() => copyToClipboard(JSON.stringify(toolData.args, null, 2), `args-${messageId}`)}
                        className="p-1 hover:bg-blue-100 rounded transition-colors"
                      >
                        {copiedStates[`args-${messageId}`] ? <Check size={14} className="text-green-600" /> : <Copy size={14} className="text-blue-600" />}
                      </button>
                    </div>
                    <pre className="text-sm text-blue-700 bg-white p-2 rounded border overflow-x-auto">
                      {JSON.stringify(toolData.args, null, 2)}
                    </pre>
                  </div>
                  
                  <div className="bg-green-50 p-3 rounded-xl">
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="font-medium text-green-800">Response</h5>
                      <button
                        onClick={() => copyToClipboard(JSON.stringify(toolData.content, null, 2), `content-${messageId}`)}
                        className="p-1 hover:bg-green-100 rounded transition-colors"
                      >
                        {copiedStates[`content-${messageId}`] ? <Check size={14} className="text-green-600" /> : <Copy size={14} className="text-green-600" />}
                      </button>
                    </div>
                    <pre className="text-sm text-green-700 bg-white p-2 rounded border overflow-x-auto max-h-64">
                      {JSON.stringify(toolData.content, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      );
    }
    if (message.type === 'error') {
      return (
        <div key={message.id} className="flex justify-start mb-6">
          <div className="flex items-start max-w-3xl">
            <div className="mr-3 p-2 bg-red-500 rounded-full">
              <Bot size={20} className="text-white" />
            </div>
            <div className="bg-red-50 border border-red-200 p-4 rounded-2xl rounded-bl-md">
              <p className="text-red-800">{message.content}</p>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-gray-200 shadow-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              {/* Company Logo */}
              <img 
                src={LOGO_URL} 
                alt="Company_Logo" 
                className="w-8 h-8 rounded-lg"
                onError={(e) => {
                  // Fallback to default Bot icon if logo fails to load
                  e.target.style.display = 'none';
                  e.target.nextElementSibling.style.display = 'flex';
                }}
              />
              {/* Fallback Bot icon */}
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl" style={{display: 'none'}}>
                <Bot size={24} className="text-white" />
              </div>
              {/* DataHarbor Logo */}
              <img 
                src={DATAHARBOR_LOGO_URL} 
                alt="DataHarbor Logo" 
                className="w-6 h-6"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
              <h1 className="text-xl font-bold text-gray-900">ConvoDB</h1>
            </div>
            
            {/* Server Selection and Settings */}
            <div className="flex items-center space-x-3">
              {/* Server Selector */}
              {/*<div className="flex items-center space-x-2">*/}
              <div className="flex items-center space-x-2 p-2 rounded-lg bg-gray-100 border border-gray-200">
                <Server size={16} className="text-gray-500" />
                {serverManagerEnabled ? (
                  <span className="text-sm font-semibold text-green-600">Server Manager Active</span>
                ) : (
                <select
                  value={activeServer}
                  onChange={(e) => handleServerSwitch(e.target.value)}
                  disabled={isServerSwitching}
                  className="px-3 py-1 border border-gray-300 rounded-lg bg-white text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50">
                  {availableServers.map((server) => (
                    <option key={server} value={server}>
                      {server}
                    </option>
                  ))}
                </select>
                )}
                {isServerSwitching && !serverManagerEnabled && (
                  <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                )}
              </div>
              
              <button
                onClick={toggleSidebar}
                className={`p-2 rounded-lg transition-all duration-200 ${
                  sidebarOpen 
                    ? 'bg-blue-100 text-blue-600 hover:bg-blue-200' 
                    : 'hover:bg-gray-100 text-gray-600'
                }`}
              >
                <Settings size={20} />
              </button>
            </div>
          </div>
        </div>
      </header>
      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar */}
        <div className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} fixed inset-y-0 left-0 z-30 w-80 bg-white/90 backdrop-blur-md border-r border-gray-200 shadow-lg transition-transform duration-300 ease-in-out`}>
          <div className="p-6 h-full overflow-y-auto">
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-2 flex items-center">
                <Settings size={18} className="mr-2" />
                Settings
              </h3>
              <div className="bg-gray-50 p-3 rounded-lg space-y-2">
                <div>
                  <p className="text-sm text-gray-600">API URL:</p>
                  <p className="text-sm font-mono text-gray-800 break-all">{API_URL}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Active Server:</p>
                  <p className="text-sm font-semibold text-blue-600">{activeServer}</p>
                </div>
              </div>
            </div>
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                <Server size={18} className="mr-2" />
                {serverManagerEnabled ? 'Connected Servers' : 'Available Servers'}
              </h3>
              {serverManagerEnabled && (
                <div className="text-xs text-center text-white bg-green-500 rounded-full px-3 py-1 mb-3">
                  Automatic Server Selection is ON
                </div>
              )}
              <div className="space-y-2">
                {availableServers.map((server) => (
                  <button
                    key={server}
                    onClick={() => handleServerSwitch(server)}
                    disabled={isServerSwitching || serverManagerEnabled} // Disable in manager mode
                    className={`w-full text-left p-3 rounded-lg border transition-all ${
                      (server === activeServer && !serverManagerEnabled) || (serverManagerEnabled && Array.isArray(activeServer) && activeServer.includes(server))
                        ? 'bg-blue-50 border-blue-200 text-blue-800'
                        : 'bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100'
                                                                                    } ${serverManagerEnabled ? 'cursor-not-allowed opacity-70' : ''}`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{server}</span>
                      {server === activeServer && (
                        <div className="w-2 h-2 bg-green-500 rounded-full" />
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </div>
            {sessionMetrics && (
  <div className="mb-6">
    <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
      <Activity size={18} className="mr-2" />
      Session Metrics
    </h3>
    <div className="bg-gradient-to-r from-green-50 to-blue-50 p-3 rounded-lg border border-green-100 space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-sm text-gray-600">Queries:</span>
        <span className="font-semibold text-green-700">{sessionMetrics.queries}</span>
      </div>
      <div className="flex justify-between items-center">
        <span className="text-sm text-gray-600">Tool Calls:</span>
        <span className="font-semibold text-blue-700">{sessionMetrics.tool_calls}</span>
      </div>
      <div className="flex justify-between items-center">
        <span className="text-sm text-gray-600">Tokens:</span>
        <span className="font-semibold text-purple-700">{sessionMetrics.total_tokens?.toLocaleString()}</span>
      </div>
      <div className="flex justify-between items-center">
        <span className="text-sm text-gray-600">Est. Cost:</span>
        <span className="font-semibold text-red-700">${sessionMetrics.estimated_cost_usd}</span>
      </div>
      <div className="flex justify-between items-center">
        <span className="text-sm text-gray-600">Errors:</span>
        <span className="font-semibold text-red-600">{sessionMetrics.errors}</span>
      </div>
      {sessionMetrics.top_tools && sessionMetrics.top_tools.length > 0 && (
        <div className="pt-2 border-t border-green-200">
          <p className="text-xs text-gray-600 mb-1">Top Tools:</p>
          {sessionMetrics.top_tools.slice(0, 3).map(([tool, count], index) => (
            <div key={tool} className="flex justify-between items-center">
              <span className="text-xs text-gray-700 truncate">{tool}</span>
              <span className="text-xs font-semibold text-blue-600">{count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  </div>
)}      
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                <Wrench size={18} className="mr-2" />
                Available Tools ({tools.length})
              </h3>
              <div className="space-y-2">
                {tools.map((tool, index) => (
                  <div key={index} className="bg-gradient-to-r from-blue-50 to-indigo-50 p-3 rounded-lg border border-blue-100">
                    <h4 className="font-medium text-blue-800 text-sm">{tool.name}</h4>
                    <p className="text-xs text-blue-600 mt-1">{tool.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
        {/* Overlay for mobile and desktop when sidebar is open */}
        {sidebarOpen && (
          <div 
            className="fixed inset-0 bg-black bg-opacity-50 z-20"
            onClick={() => setSidebarOpen(false)}
          />
        )}
        {/* Main Content */}
        <div className="flex-1 flex flex-col w-full">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 md:p-6">
            <div className="max-w-4xl mx-auto">
              {messages.length === 0 ? (
                <div className="text-center py-12">
                  <div className="mb-4 mx-auto w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                    <Bot size={32} className="text-white" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-700 mb-2">Connect and chat directly with your data via the MCP platform!</h2>
                  <p className="text-gray-500">Start a conversation by typing your query below.</p>
                  <p className="text-sm text-blue-600 mt-2">Currently connected to: <strong>{activeServer}</strong></p>
                </div>
              ) : (
                messages.map(renderMessage)
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
          {/* Input Form */}
          <div className="border-t border-gray-200 bg-white/80 backdrop-blur-md p-4">
            <div className="max-w-4xl mx-auto">
              <div className="flex space-x-3">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit(e);
                    }
                  }}
                  placeholder="Enter your query here..."
                  disabled={isLoading || isServerSwitching}
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                />
                <button
                  onClick={handleSubmit}
                  disabled={isLoading || !query.trim() || isServerSwitching}
                  className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl hover:from-blue-600 hover:to-purple-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center space-x-2"
                >
                  {isLoading ? (
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Send size={20} />
                  )}
                  <span className="hidden sm:inline">Send</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
export default MCPChatbot;
