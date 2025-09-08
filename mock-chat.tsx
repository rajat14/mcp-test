/**
 * AI Chat Panel Component
 * Integrates with MCP server for intelligent lineage analysis
 */

import { Send, User, X, ChevronRight, HelpCircle, Bot } from 'lucide-react';
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { ChatSettings, type ChatConfig } from './ChatSettings';
import { SampleQuestionsModal } from './SampleQuestionsModal';
import { McpToolsModal } from './McpToolsModal';
import { ExecutionDetails } from './ExecutionDetails';
import { buildLineageContext } from '@/utils/llm-client';
import { dataharborApi } from '@/utils/dataharbor-api';
import { MCPExecuteRequest, MCPExecuteResponse } from '@/types/dataharbor';

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'tool';
  content: string;
  timestamp: Date;
  toolUsed?: string;
  executionDetails?: {
    execution_plan?: any;
    tools_used?: string[];
    results?: Record<string, any>;
  };
}

interface ChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
  selectedNode?: string;
  onNodeHighlight?: (nodeId: string) => void;
  isLineagePage?: boolean;
  lineageDatabase?: string;
  lineageTable?: string;
  // New props for LLM integration
  reactFlowNodes?: any[];
  reactFlowEdges?: any[];
  selectedFlowNode?: any;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({
  isOpen,
  onClose,
  selectedNode,
  onNodeHighlight,
  isLineagePage = false,
  lineageDatabase = '',
  lineageTable = '',
  reactFlowNodes = [],
  reactFlowEdges = [],
  selectedFlowNode
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSampleQuestions, setShowSampleQuestions] = useState(false);
  const [hasLineageContext, setHasLineageContext] = useState(isLineagePage && !!(lineageDatabase || lineageTable));
  const [chatConfig, setChatConfig] = useState<ChatConfig>(() => {
    // Load config from localStorage
    const stored = localStorage.getItem('dataharbor-chat-config');
    if (stored) {
      try {
        return JSON.parse(stored);
      } catch (e) {
        console.warn('Failed to parse stored chat config:', e);
      }
    }
    return {
      apiKey: '',
      model: 'gpt-4',
      provider: 'openai',
      useRealBackend: true
    };
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    // Add welcome message on mount
    if (messages.length === 0) {
      addMessage('assistant', 
        '🤖 Hi! I\'m CompassAI, your AI guide through the data landscape. I can help you analyze data lineage, node relationships, field connections, and answer questions about what you\'re seeing. Try asking me questions like "How many nodes are visible?" or "What about field lineage?"'
      );
    }
  }, []);

  useEffect(() => {
    setHasLineageContext(isLineagePage && !!(lineageDatabase || lineageTable));
  }, [isLineagePage, lineageDatabase, lineageTable]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleConfigChange = (newConfig: ChatConfig) => {
    const prevConfig = chatConfig;
    setChatConfig(newConfig);
    localStorage.setItem('dataharbor-chat-config', JSON.stringify(newConfig));
    
    // Update welcome message based on configuration changes
    if (newConfig.useRealBackend && !prevConfig.useRealBackend) {
      addMessage('assistant', 
        '🚀 Real backend integration enabled! I can now use the DataHarbor MCP backend for intelligent responses. Try asking me complex questions about your data pipeline!'
      );
    } else if (!newConfig.useRealBackend && prevConfig.useRealBackend) {
      addMessage('assistant', 
        '📱 Switched to mock mode. I\'ll provide simulated responses based on the current lineage view.'
      );
    } else if (newConfig.apiKey && !prevConfig.apiKey) {
      addMessage('assistant', 
        '✅ Great! Your API key is configured. I can now help you analyze the lineage data. Try asking me questions like "How many nodes are visible?" or "Which fields have lineage connections?"'
      );
    }
  };

  const addMessage = (
    type: ChatMessage['type'], 
    content: string, 
    toolUsed?: string, 
    executionDetails?: ChatMessage['executionDetails']
  ) => {
    const message: ChatMessage = {
      id: Date.now().toString(),
      type,
      content,
      timestamp: new Date(),
      toolUsed,
      executionDetails
    };
    setMessages(prev => [...prev, message]);
  };

  const callRealBackend = async (userMessage: string): Promise<{
    response: string;
    executionDetails?: ChatMessage['executionDetails'];
  }> => {
    try {
      const request: MCPExecuteRequest = {
        user_query: userMessage
      };
      
      const response: MCPExecuteResponse = await dataharborApi.mcpExecute(request);
      
      // Handle the response structure provided by the user
      if (response && typeof response === 'object') {
        const typedResponse = response as any;
        
        // Extract execution details
        const executionDetails: ChatMessage['executionDetails'] = {
          execution_plan: typedResponse.execution_plan,
          tools_used: typedResponse.tools_used,
          results: typedResponse.results
        };
        
        if (typedResponse.natural_response) {
          return {
            response: typedResponse.natural_response,
            executionDetails
          };
        }
        
        // Fallback to content if natural_response is not available
        if (response.content && Array.isArray(response.content)) {
          return {
            response: response.content.map((item: any) => 
              typeof item === 'string' ? item : item.text || JSON.stringify(item)
            ).join('\n'),
            executionDetails
          };
        }
        
        // If response has error
        if (response.isError) {
          return {
            response: `❌ Backend error: ${JSON.stringify(response)}`,
            executionDetails
          };
        }
        
        // Fallback: stringify the response
        return {
          response: JSON.stringify(response, null, 2),
          executionDetails
        };
      }
      
      return {
        response: 'No response received from backend.'
      };
      
    } catch (error) {
      console.error('Real backend error:', error);
      return {
        response: `❌ Failed to connect to DataHarbor backend: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  };

  const callLLM = async (userMessage: string): Promise<{
    response: string;
    executionDetails?: ChatMessage['executionDetails'];
  }> => {
    // Use real backend if enabled
    if (chatConfig.useRealBackend) {
      return await callRealBackend(userMessage);
    }
    
    // Simulate API delay for mock responses
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

    const message = userMessage.toLowerCase();
    
    // Build lineage context if we're on a lineage page with data
    let context = null;
    if (hasLineageContext && reactFlowNodes.length > 0) {
      context = buildLineageContext(
        reactFlowNodes,
        reactFlowEdges,
        lineageDatabase,
        lineageTable,
        selectedFlowNode
      );
    }

    // Mock responses based on user intent and available context
    if (message.includes('how many') && message.includes('node')) {
      if (context) {
        return {
          response: `I can see **${context.summary.totalNodes} nodes** in the current lineage view for ${context.currentView.database}.${context.currentView.table}. The breakdown by type is:\n\n${Object.entries(context.summary.nodeTypes).map(([type, count]) => `• **${type}**: ${count}`).join('\n')}`
        };
      } else {
        return {
          response: 'I don\'t see any lineage data loaded currently. Please navigate to a lineage view to see node information.'
        };
      }
    }

    if (message.includes('field') && (message.includes('lineage') || message.includes('connection'))) {
      if (context) {
        const totalFields = context.nodeDetails.reduce((sum, node) => sum + node.totalFields, 0);
        const fieldsWithLineage = context.nodeDetails.reduce((sum, node) => sum + node.fieldsWithLineage, 0);
        return {
          response: `In the current lineage view, there are **${totalFields} total fields** across all nodes, with **${fieldsWithLineage} fields** that have lineage connections. That's about ${Math.round((fieldsWithLineage / totalFields) * 100)}% of fields with tracked lineage.`
        };
      } else {
        return {
          response: 'To analyze field lineage, please navigate to a specific table\'s lineage view first.'
        };
      }
    }

    if (message.includes('data flow') || message.includes('path')) {
      if (context) {
        const jobNodes = context.nodeDetails.filter(n => n.type === 'job');
        const datasetNodes = context.nodeDetails.filter(n => n.type === 'dataset');
        return {
          response: `The data flow shows **${datasetNodes.length} datasets** connected through **${jobNodes.length} processing jobs** with **${context.summary.totalEdges} total connections**. The main job node appears to be the central transformation point in this pipeline.`
        };
      } else {
        return {
          response: 'I\'d need to see the lineage diagram to trace the data flow paths for you.'
        };
      }
    }

    if (message.includes('expand') || message.includes('collapse')) {
      if (context) {
        const expandedNodes = context.nodeDetails.filter(n => n.isExpanded).length;
        return {
          response: `Currently **${expandedNodes} nodes** are expanded out of ${context.summary.totalNodes} total nodes. You can expand nodes to see more details about their fields and connections.`
        };
      } else {
        return {
          response: 'Node expansion controls are available when viewing lineage diagrams.'
        };
      }
    }

    if (message.includes('impact') || message.includes('affect')) {
      return {
        response: 'For impact analysis, I can help you understand:\n\n• **Downstream effects** - what would be affected if this data source changes\n• **Upstream dependencies** - what feeds into this dataset\n• **Field-level impacts** - which specific columns would be affected\n\nTry asking about a specific table or dataset!'
      };
    }

    if (message.includes('sql') || message.includes('query')) {
      if (context) {
        const tableNodes = context.nodeDetails.filter(n => n.type === 'table');
        if (tableNodes.length >= 2) {
          return {
            response: `I can see ${tableNodes.length} tables in this lineage. To generate SQL joins, I\'d need to analyze the field connections between:\n\n${tableNodes.map(t => `• **${t.label}** (${t.totalFields} fields)`).join('\n')}\n\nWould you like me to suggest join patterns based on the lineage connections?`
          };
        }
      }
      return {
        response: 'SQL generation works best when viewing lineage with multiple connected tables. Navigate to a complex lineage view to see join suggestions.'
      };
    }

    if (message.includes('transform') || message.includes('column') || message.includes('field')) {
      if (context) {
        const jobNodes = context.nodeDetails.filter(n => n.type === 'job');
        const tableNodes = context.nodeDetails.filter(n => n.type === 'table');
        
        if (message.includes('transform') && jobNodes.length > 0) {
          return {
            response: `I can see transformations happening in the **${jobNodes[0].label}** job node. Based on the field connections, here are some common transformations I observe:\n\n• **Data type conversions** - timestamp to date, string to numeric\n• **Field renaming** - customer_id → cust_id, order_date → created_at\n• **Aggregations** - SUM(amount), COUNT(orders), AVG(rating)\n• **Joins** - combining customer data with order history\n• **Filtering** - WHERE status = 'active' AND date >= '2024-01-01'\n\nThe lineage shows ${context.summary.fieldConnections} field-level connections tracking these transformations.`
          };
        }
        
        if (message.includes('column') || message.includes('field')) {
          const specificColumns = ['customer_id', 'order_id', 'amount', 'status', 'created_at', 'email', 'phone'];
          const mentionedColumn = specificColumns.find(col => message.includes(col));
          
          if (mentionedColumn) {
            return {
              response: `The **${mentionedColumn}** column flows through several transformations:\n\n1. **Source**: Raw ${mentionedColumn} from upstream table\n2. **Validation**: NULL checks and data quality rules\n3. **Transformation**: ${mentionedColumn === 'amount' ? 'Currency conversion and rounding' : mentionedColumn === 'created_at' ? 'Timezone normalization' : 'Format standardization'}\n4. **Target**: Clean ${mentionedColumn} in downstream datasets\n\nYou can trace this by following the field-level connections (cyan lines) in the lineage view.`
            };
          } else {
            return `Field transformations in this lineage include:\n\n• **${Math.floor(context.summary.fieldConnections * 0.6)} direct mappings** - 1:1 field copies\n• **${Math.floor(context.summary.fieldConnections * 0.3)} calculated fields** - derived values and aggregations\n• **${Math.floor(context.summary.fieldConnections * 0.1)} split/merge operations** - combining or splitting data\n\nTry asking about a specific column like "customer_id" or "amount" for detailed transformation logic!`;
          }
        }
      }
      return 'To see column transformations, navigate to a lineage view with job nodes. I can then trace how fields are transformed between source and target tables.';
    }

    if (message.includes('anomal') || message.includes('issue') || message.includes('problem')) {
      if (context) {
        const issues = [];
        const nodesWithoutFields = context.nodeDetails.filter(n => n.totalFields === 0);
        const nodesWithoutLineage = context.nodeDetails.filter(n => n.fieldsWithLineage === 0 && n.totalFields > 0);
        
        if (nodesWithoutFields.length > 0) {
          issues.push(`• ${nodesWithoutFields.length} nodes have no field information`);
        }
        if (nodesWithoutLineage.length > 0) {
          issues.push(`• ${nodesWithoutLineage.length} nodes have fields but no lineage connections`);
        }
        
        if (issues.length > 0) {
          return `I found some potential issues in the current lineage:\n\n${issues.join('\n')}\n\nThese might indicate incomplete metadata or configuration issues.`;
        } else {
          return '✅ The lineage looks healthy! All nodes have appropriate field information and connections.';
        }
      }
      return 'I\'d need to analyze a lineage view to detect potential anomalies or issues.';
    }

    if (message.includes('quality') || message.includes('validation') || message.includes('clean')) {
      if (context) {
        return `Data quality checks I can see in this lineage:\n\n• **Null validation** - ensuring required fields are populated\n• **Format validation** - email, phone, date format checks\n• **Range validation** - amount > 0, age between 18-120\n• **Referential integrity** - foreign key constraints\n• **Duplicate detection** - identifying and handling duplicates\n\nThe job nodes typically handle these validations before passing clean data downstream. ${context.summary.fieldConnections} field connections suggest comprehensive data quality pipelines.`;
      }
      return 'Data quality analysis works best when viewing lineage with transformation jobs. Navigate to a lineage view to see quality checks and validations.';
    }

    if (message.includes('join') || message.includes('merge') || message.includes('combine')) {
      if (context) {
        const tableNodes = context.nodeDetails.filter(n => n.type === 'table');
        if (tableNodes.length >= 2) {
          return `I can see potential joins between tables in this lineage:\n\n• **${tableNodes[0].label}** ⟷ **${tableNodes[1].label}** via customer_id\n• **Primary keys**: id, customer_id, order_id\n• **Join types**: INNER JOIN (active records), LEFT JOIN (include nulls)\n• **Common fields**: created_at, updated_at, status\n\nThe field-level connections (cyan lines) show exactly which columns are being joined. Would you like me to suggest the SQL for these joins?`;
        }
      }
      return 'Join analysis requires multiple connected tables. Navigate to a complex lineage view with multiple tables to see join patterns.';
    }

    if (message.includes('aggregate') || message.includes('sum') || message.includes('count') || message.includes('avg')) {
      if (context) {
        const jobNodes = context.nodeDetails.filter(n => n.type === 'job');
        if (jobNodes.length > 0) {
          return `Aggregation operations in **${jobNodes[0].label}**:\n\n• **SUM(order_amount)** - total revenue per customer\n• **COUNT(orders)** - number of orders per customer\n• **AVG(rating)** - average customer satisfaction\n• **MAX(order_date)** - most recent order date\n• **MIN(first_order)** - customer acquisition date\n\nThese aggregations create summary tables from detailed transaction data. The lineage shows how raw events become analytics-ready datasets.`;
        }
      }
      return 'Aggregation analysis works best with job nodes that perform calculations. Navigate to a lineage with transformation jobs to see aggregation patterns.';
    }

    // General responses
    const responses = [
      'I can help you analyze data lineage! Try asking me about:\n\n• **Node counts and types**\n• **Field transformations**\n• **Column mappings**\n• **Data quality checks**\n• **Join patterns**\n• **Aggregations**',
      
      'As your CompassAI assistant, I can provide insights about:\n\n• **Lineage visualization** - understanding node relationships\n• **Field mapping** - tracking column-level dependencies\n• **Transformation logic** - how data is processed and changed\n• **Data quality** - validation rules and cleansing steps\n• **Impact analysis** - downstream and upstream effects',
      
      context ? 
        `Based on the current lineage view of **${context.currentView.database}.${context.currentView.table}**, I can see ${context.summary.totalNodes} nodes with ${context.summary.totalEdges} connections. Try asking about transformations, field mappings, or data quality!` :
        'I\'m ready to help with lineage analysis! Navigate to a lineage view and I can provide detailed insights about transformations, data flow, and column mappings.',
    ];

    return {
      response: responses[Math.floor(Math.random() * responses.length)]
    };
  };

  const handleQuestionSelect = (question: string) => {
    setInput(question);
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    addMessage('user', userMessage);
    setIsLoading(true);

    try {
      const result = await callLLM(userMessage);
      addMessage('assistant', result.response, undefined, result.executionDetails);
      
      // Extract node names from response for highlighting
      if (onNodeHighlight) {
        const nodeMatches = result.response.match(/\b\w+_\w+\b/g);
        if (nodeMatches) {
          nodeMatches.forEach(node => onNodeHighlight(node));
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      addMessage('assistant', '❌ Sorry, I encountered an error processing your request. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Removed formatMessage function - using ReactMarkdown instead

  if (!isOpen) return null;

  return (
    <div className="h-full flex flex-col bg-white dark:bg-slate-800">
      <div className="flex flex-row items-center justify-between p-4 border-b border-gray-200 dark:border-slate-700 flex-shrink-0">
        <div className="text-lg font-semibold flex items-center gap-2">
          <Bot className="w-5 h-5" />
          CompassAI
          <span className="text-xs font-normal text-muted-foreground">Beta</span>
        </div>
        <div className="flex items-center gap-2">
          {!chatConfig.useRealBackend && (
            <Badge variant="secondary" className="text-xs">
              Mock Mode
            </Badge>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowSampleQuestions(true)}
            className="h-8 w-8 p-0 hover:bg-slate-100 dark:hover:bg-slate-700"
            title="Sample Questions"
          >
            <HelpCircle className="h-4 w-4" />
          </Button>
          <McpToolsModal />
          <ChatSettings config={chatConfig} onConfigChange={handleConfigChange} />
          <Button variant="ghost" size="sm" onClick={onClose}>
            ✕
          </Button>
        </div>
      </div>
      
      <div className="flex-1 flex flex-col min-h-0">
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4 pb-20">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex w-full overflow-hidden ${
                  message.type === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div className={`flex items-start space-x-3 w-full max-w-none ${
                  message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                }`}>
                  <div className={`flex items-center justify-center w-8 h-8 rounded-full flex-shrink-0 ${
                    message.type === 'user' 
                      ? 'bg-orange-500' 
                      : 'bg-gradient-to-br from-blue-500 to-purple-600'
                  }`}>
                    {message.type === 'user' ? (
                      <User className="w-4 h-4 text-white" />
                    ) : (
                      <Bot className="w-4 h-4 text-white" />
                    )}
                  </div>
                  
                  <div className={`rounded-lg px-4 py-3 flex-1 min-w-0 ${
                    message.type === 'user'
                      ? 'bg-orange-500'
                      : 'bg-slate-100 dark:bg-slate-700'
                  }`}>
                    {message.toolUsed && (
                      <Badge variant="outline" className="text-xs mb-2">
                        {message.toolUsed}
                      </Badge>
                    )}
                    
                    {/* Show execution details before the response for assistant messages */}
                    {message.type === 'assistant' && message.executionDetails && (
                      <div className="mb-3">
                        <ExecutionDetails
                          executionPlan={message.executionDetails.execution_plan}
                          toolsUsed={message.executionDetails.tools_used}
                          results={message.executionDetails.results}
                        />
                      </div>
                    )}
                    
                    <div className={`prose prose-sm max-w-none ${
                      message.type === 'user' 
                        ? 'text-white prose-invert' 
                        : 'text-slate-900 dark:text-white dark:prose-invert'
                    }`}>
                      {message.type === 'user' ? (
                        // For user messages, keep as plain text
                        <p className="whitespace-pre-wrap break-words">{message.content}</p>
                      ) : (
                        // For assistant messages, use ReactMarkdown
                        <div className="markdown-content">
                          <ReactMarkdown
                            components={{
                              // Custom styling for code blocks
                              pre: ({ children }) => (
                                <pre className="bg-slate-900 text-green-400 p-3 rounded mt-2 mb-2 overflow-x-auto whitespace-pre-wrap break-words max-w-full">
                                  {children}
                                </pre>
                              ),
                              code: ({ children, className, ...props }) => {
                                const isInline = !className?.includes('language-');
                                return isInline ? (
                                  <code className="bg-slate-200 dark:bg-slate-700 px-1 py-0.5 rounded text-sm break-words" {...props}>
                                    {children}
                                  </code>
                                ) : (
                                  <code className={`${className} break-words`} {...props}>{children}</code>
                                );
                              }
                            }}
                          >
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="flex items-start space-x-3">
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                  <div className="bg-slate-100 dark:bg-slate-700 rounded-lg px-4 py-3">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Scroll target */}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>
        
        {hasLineageContext && (
          <div className="px-4 pb-2">
            <div className="flex items-center gap-2 bg-orange-100 dark:bg-orange-900/20 text-orange-700 dark:text-orange-300 px-3 py-1.5 rounded-full text-sm inline-flex">
              <span className="font-medium truncate max-w-[120px]">
                {lineageDatabase || 'Database'}
              </span>
              {lineageTable && (
                <>
                  <ChevronRight className="w-3 h-3 text-orange-500 dark:text-orange-400 flex-shrink-0" />
                  <span className="font-medium truncate max-w-[120px]">
                    {lineageTable}
                  </span>
                </>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setHasLineageContext(false)}
                className="h-4 w-4 p-0 text-orange-600 dark:text-orange-400 hover:text-orange-800 dark:hover:text-orange-200 ml-1 flex-shrink-0"
              >
                <X className="h-3 w-3" />
              </Button>
            </div>
          </div>
        )}
        
      </div>
      
      <div className="p-4 border-t border-gray-200 dark:border-slate-700">
          <form 
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
            className="flex space-x-2"
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              placeholder={
                hasLineageContext
                  ? selectedNode 
                    ? lineageDatabase && lineageTable
                      ? `Ask about ${selectedNode} or ${lineageDatabase}.${lineageTable} lineage...`
                      : `Ask about ${selectedNode} or the lineage view...`
                    : lineageDatabase && lineageTable
                      ? `Ask about ${lineageDatabase}.${lineageTable} lineage, dependencies, or data flow...`
                      : "Ask about the lineage view, data flow, or dependencies..."
                  : selectedNode 
                  ? `Ask about ${selectedNode}...`
                  : "Ask CompassAI about your data pipeline..."
              }
              disabled={isLoading}
              className="flex-1 min-h-[42px] max-h-[120px] resize-y rounded-md bg-white dark:bg-slate-800 border border-gray-300 dark:border-slate-600 text-black dark:text-white placeholder-gray-500 dark:placeholder-gray-400 px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              rows={1}
            />
            <Button 
              type="submit"
              disabled={isLoading || !input.trim()}
              className="bg-orange-500 hover:bg-orange-600 text-white"
            >
              <Send className="w-4 h-4" />
            </Button>
          </form>
      </div>
      
      <SampleQuestionsModal
        open={showSampleQuestions}
        onOpenChange={setShowSampleQuestions}
        onQuestionSelect={handleQuestionSelect}
      />
    </div>
  );
};
