import { useState, useEffect, useRef } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type {
  AgentSummary,
  LLMStatus,
  NarrativeResponse,
  DecisionExplainResponse,
  AgentDetail,
} from '../../../types';

type Tab = 'interview' | 'narratives' | 'decisions' | 'settings';

export function InterviewView() {
  const { activeSessionId } = useSimulationStore();
  const [tab, setTab] = useState<Tab>('interview');
  const [llmStatus, setLlmStatus] = useState<LLMStatus | null>(null);

  // Provider settings
  const [provider, setProvider] = useState<string>('anthropic');
  const [selectedModel, setSelectedModel] = useState<string>('');

  const refreshStatus = () => {
    api.getLLMStatus().then(setLlmStatus).catch(() => {});
  };

  useEffect(() => {
    refreshStatus();
  }, []);

  // Auto-select provider based on availability
  useEffect(() => {
    if (!llmStatus) return;
    if (!llmStatus.providers.anthropic.available && llmStatus.providers.ollama.available) {
      setProvider('ollama');
      const models = llmStatus.providers.ollama.models ?? [];
      if (models.length > 0 && !selectedModel) setSelectedModel(models[0]);
    }
  }, [llmStatus, selectedModel]);

  const unavailable = llmStatus && !llmStatus.available;

  const tabs: { key: Tab; label: string }[] = [
    { key: 'interview', label: 'Interview' },
    { key: 'narratives', label: 'Narratives' },
    { key: 'decisions', label: 'Decision Explorer' },
    { key: 'settings', label: 'Settings' },
  ];

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-gray-100">Agent Interview</h1>

      {unavailable && (
        <div className="rounded-lg border border-amber-700 bg-amber-900/30 px-4 py-3 text-sm text-amber-300">
          No LLM provider available. Set <code className="font-mono text-amber-200">ANTHROPIC_API_KEY</code> or start{' '}
          <code className="font-mono text-amber-200">Ollama</code> locally to enable LLM features.
        </div>
      )}

      {!activeSessionId && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-8 text-center text-gray-500">
          Create or select a simulation session from Mission Control to use LLM features.
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-800">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              tab === t.key
                ? 'border-b-2 border-blue-500 text-gray-100'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'interview' && activeSessionId && (
        <InterviewTab sessionId={activeSessionId} provider={provider} model={selectedModel} />
      )}
      {tab === 'narratives' && activeSessionId && (
        <NarrativesTab sessionId={activeSessionId} provider={provider} model={selectedModel} />
      )}
      {tab === 'decisions' && activeSessionId && (
        <DecisionExplorerTab sessionId={activeSessionId} provider={provider} model={selectedModel} />
      )}
      {tab === 'settings' && (
        <SettingsTab
          llmStatus={llmStatus}
          provider={provider}
          setProvider={setProvider}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          onStatusRefresh={refreshStatus}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Settings Tab
// ---------------------------------------------------------------------------

function SettingsTab({
  llmStatus,
  provider,
  setProvider,
  selectedModel,
  setSelectedModel,
  onStatusRefresh,
}: {
  llmStatus: LLMStatus | null;
  provider: string;
  setProvider: (p: string) => void;
  selectedModel: string;
  setSelectedModel: (m: string) => void;
  onStatusRefresh: () => void;
}) {
  const anthropicModels = llmStatus?.providers.anthropic.models ?? [];
  const ollamaModels = llmStatus?.providers.ollama.models ?? [];
  const ollamaBaseUrl = llmStatus?.providers.ollama.base_url ?? '';

  const [apiKey, setApiKey] = useState('');
  const [keyStatus, setKeyStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [keyError, setKeyError] = useState('');
  const [customOllamaUrl, setCustomOllamaUrl] = useState(ollamaBaseUrl);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [testing, setTesting] = useState(false);

  // Sync Ollama URL from status
  useEffect(() => {
    if (ollamaBaseUrl && !customOllamaUrl) setCustomOllamaUrl(ollamaBaseUrl);
  }, [ollamaBaseUrl]);

  const handleSaveKey = async () => {
    if (!apiKey.trim()) return;
    setKeyStatus('saving');
    setKeyError('');
    try {
      await api.setLLMApiKey(apiKey.trim());
      setKeyStatus('success');
      setApiKey('');
      onStatusRefresh();
    } catch (err: unknown) {
      setKeyStatus('error');
      setKeyError(err instanceof Error ? err.message : 'Failed to set API key');
    }
  };

  const handleClearKey = async () => {
    try {
      await api.clearLLMApiKey();
      setKeyStatus('idle');
      onStatusRefresh();
    } catch {
      // ignore
    }
  };

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const result = await api.testLLMConnection({
        provider,
        model: selectedModel || undefined,
        ollama_base_url: provider === 'ollama' ? customOllamaUrl || undefined : undefined,
      });
      setTestResult(result);
      if (result.success) onStatusRefresh();
    } catch (err: unknown) {
      setTestResult({ success: false, message: err instanceof Error ? err.message : 'Connection test failed' });
    }
    setTesting(false);
  };

  const handleSaveOllamaUrl = async () => {
    try {
      await api.setOllamaUrl(customOllamaUrl);
      onStatusRefresh();
    } catch {
      // handled by test connection
    }
  };

  const models = provider === 'anthropic' ? anthropicModels : ollamaModels;

  return (
    <div className="max-w-2xl space-y-6">
      {/* Provider Selection - Card Layout */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-5">
        <h2 className="mb-4 text-lg font-semibold text-gray-200">LLM Provider</h2>
        <div className="grid grid-cols-2 gap-3">
          {/* Anthropic Card */}
          <button
            onClick={() => {
              setProvider('anthropic');
              const am = llmStatus?.providers.anthropic.models ?? [];
              if (am.length > 0) setSelectedModel(am[0]);
              else setSelectedModel('');
              setTestResult(null);
            }}
            className={`relative rounded-lg border-2 p-4 text-left transition-all ${
              provider === 'anthropic'
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-gray-200">Anthropic</span>
              <span className={`h-5 w-5 rounded-full border-2 flex items-center justify-center ${
                provider === 'anthropic' ? 'border-blue-500 bg-blue-500' : 'border-gray-600'
              }`}>
                {provider === 'anthropic' && (
                  <svg className="h-3 w-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
              </span>
            </div>
            <p className="mt-1 text-xs text-gray-500">Claude models from Anthropic</p>
            {llmStatus?.providers.anthropic.available && (
              <span className="absolute top-2 right-8 h-2 w-2 rounded-full bg-green-500" />
            )}
          </button>

          {/* Ollama Card */}
          <button
            onClick={() => {
              setProvider('ollama');
              const om = llmStatus?.providers.ollama.models ?? [];
              if (om.length > 0) setSelectedModel(om[0]);
              else setSelectedModel('');
              setTestResult(null);
            }}
            className={`relative rounded-lg border-2 p-4 text-left transition-all ${
              provider === 'ollama'
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-gray-200">Ollama</span>
              <span className={`h-5 w-5 rounded-full border-2 flex items-center justify-center ${
                provider === 'ollama' ? 'border-blue-500 bg-blue-500' : 'border-gray-600'
              }`}>
                {provider === 'ollama' && (
                  <svg className="h-3 w-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
              </span>
            </div>
            <p className="mt-1 text-xs text-gray-500">Local inference (no API key required)</p>
            {llmStatus?.providers.ollama.available && (
              <span className="absolute top-2 right-8 h-2 w-2 rounded-full bg-green-500" />
            )}
          </button>
        </div>
      </div>

      {/* Anthropic API Key */}
      {provider === 'anthropic' && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-5">
          <h2 className="mb-3 text-sm font-semibold text-gray-300">API Key</h2>
          {llmStatus?.providers.anthropic.has_key ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="rounded bg-green-900 px-2 py-0.5 text-xs text-green-300">Configured</span>
                <span className="text-xs text-gray-500">API key is active</span>
              </div>
              <button
                onClick={handleClearKey}
                className="rounded-md border border-gray-700 px-3 py-1.5 text-xs text-gray-400 hover:bg-gray-800 hover:text-gray-200"
              >
                Clear
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-xs text-gray-500">
                Enter your API key (in-memory only) or set <code className="text-gray-400">ANTHROPIC_API_KEY</code> in <code className="text-gray-400">.env</code>
              </p>
              <div className="flex gap-2">
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => { setApiKey(e.target.value); setKeyStatus('idle'); }}
                  placeholder="sk-ant-..."
                  className="flex-1 rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 placeholder:text-gray-600"
                />
                <button
                  onClick={handleSaveKey}
                  disabled={!apiKey.trim() || keyStatus === 'saving'}
                  className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
                >
                  {keyStatus === 'saving' ? 'Saving...' : 'Set Key'}
                </button>
              </div>
              {keyStatus === 'success' && <p className="text-xs text-green-400">API key set successfully.</p>}
              {keyStatus === 'error' && <p className="text-xs text-red-400">{keyError}</p>}
            </div>
          )}
        </div>
      )}

      {/* Model Selection - both providers */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-5">
        <h2 className="mb-3 text-sm font-semibold text-gray-300">Model</h2>
        {models.length > 0 ? (
          <select
            className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2.5 text-sm text-gray-200"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        ) : (
          <p className="text-sm text-gray-500">
            {provider === 'anthropic'
              ? 'Set an API key to see available models.'
              : 'No models found. Start Ollama and pull a model (e.g. ollama pull gemma3:27b).'}
          </p>
        )}
      </div>

      {/* Ollama Base URL */}
      {provider === 'ollama' && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-5">
          <h2 className="mb-3 text-sm font-semibold text-gray-300">Ollama Base URL</h2>
          <input
            type="text"
            value={customOllamaUrl}
            onChange={(e) => setCustomOllamaUrl(e.target.value)}
            onBlur={handleSaveOllamaUrl}
            placeholder="http://localhost:11434"
            className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2.5 text-sm text-gray-200 placeholder:text-gray-600"
          />
          <p className="mt-1.5 text-xs text-gray-600">URL where Ollama is running locally</p>
        </div>
      )}

      {/* Test Connection + Status */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleTestConnection}
          disabled={testing}
          className="rounded-md border border-gray-600 bg-gray-800 px-5 py-2.5 text-sm font-medium text-gray-200 hover:bg-gray-700 disabled:opacity-40"
        >
          {testing ? 'Testing...' : 'Test Connection'}
        </button>
        <button
          onClick={onStatusRefresh}
          className="rounded-md bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700"
        >
          Done
        </button>
      </div>

      {testResult && (
        <div className={`rounded-lg border px-4 py-3 text-sm ${
          testResult.success
            ? 'border-green-800 bg-green-900/30 text-green-300'
            : 'border-red-800 bg-red-900/30 text-red-300'
        }`}>
          {testResult.message}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Interview Tab
// ---------------------------------------------------------------------------

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

function InterviewTab({ sessionId, provider, model }: { sessionId: string; provider: string; model: string }) {
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [search, setSearch] = useState('');
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Historical interview mode
  const [historicalMode, setHistoricalMode] = useState(false);
  const [targetGeneration, setTargetGeneration] = useState(0);
  const [genRange, setGenRange] = useState<{ birth_generation: number; last_generation: number } | null>(null);

  useEffect(() => {
    api.listAgents(sessionId, { alive_only: false, page_size: 200 })
      .then((res) => setAgents(res.agents))
      .catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Fetch generation range when agent is selected and historical mode is on
  useEffect(() => {
    if (!selectedAgent || !historicalMode) {
      setGenRange(null);
      return;
    }
    api.getAgentGenerationRange(sessionId, selectedAgent)
      .then((range) => {
        setGenRange(range);
        setTargetGeneration(range.last_generation);
      })
      .catch(() => setGenRange(null));
  }, [sessionId, selectedAgent, historicalMode]);

  const filteredAgents = agents.filter((a) =>
    a.has_decisions && (
      a.name.toLowerCase().includes(search.toLowerCase()) ||
      a.id.toLowerCase().includes(search.toLowerCase())
    )
  );

  const handleAsk = async () => {
    if (!selectedAgent || !question.trim()) return;
    const q = question.trim();
    setQuestion('');
    setMessages((prev) => [...prev, { role: 'user', content: q }]);
    setLoading(true);

    try {
      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      let resp;
      if (historicalMode && genRange) {
        resp = await api.interviewAgentHistorical(
          sessionId, selectedAgent, q, targetGeneration, history, provider, model || undefined,
        );
      } else {
        resp = await api.interviewAgent(sessionId, selectedAgent, q, history, provider, model || undefined);
      }
      setMessages((prev) => [...prev, { role: 'assistant', content: resp.response }]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Interview request failed';
      setMessages((prev) => [...prev, { role: 'assistant', content: `[Error: ${msg}]` }]);
    }
    setLoading(false);
  };

  const handleSelectAgent = (id: string) => {
    setSelectedAgent(id);
    setMessages([]);
  };

  const selectedAgentObj = agents.find((a) => a.id === selectedAgent);

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
      {/* Agent Selector */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-3">
        <input
          placeholder="Search agents..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="mb-2 w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-gray-200 placeholder:text-gray-600"
        />
        <div className="max-h-96 space-y-1 overflow-y-auto">
          {filteredAgents.map((a) => (
            <button
              key={a.id}
              onClick={() => handleSelectAgent(a.id)}
              className={`w-full rounded px-2 py-1.5 text-left text-sm transition-colors ${
                selectedAgent === a.id
                  ? 'bg-blue-600/20 text-blue-300'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`}
            >
              <div className="font-medium">{a.name}</div>
              <div className="text-xs text-gray-600">
                {a.processing_region} | age {a.age}
                {!a.is_alive && ' (dead)'}
              </div>
            </button>
          ))}
          {filteredAgents.length === 0 && (
            <div className="py-4 text-center text-xs text-gray-600">No agents found</div>
          )}
        </div>
      </div>

      {/* Chat Panel */}
      <div className="lg:col-span-3 flex flex-col rounded-lg border border-gray-800 bg-gray-900">
        {!selectedAgent ? (
          <div className="flex flex-1 items-center justify-center p-8 text-gray-600">
            Select an agent to begin the interview
          </div>
        ) : (
          <>
            <div className="border-b border-gray-800 px-4 py-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-300">
                  Interviewing: {selectedAgentObj?.name ?? selectedAgent}
                  {historicalMode && genRange && (
                    <span className="ml-2 text-xs text-amber-400">
                      at Generation {targetGeneration}, age {targetGeneration - genRange.birth_generation}
                    </span>
                  )}
                </span>
                <label className="flex items-center gap-2 text-xs text-gray-500 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={historicalMode}
                    onChange={(e) => {
                      setHistoricalMode(e.target.checked);
                      setMessages([]);
                    }}
                    className="rounded border-gray-600 bg-gray-800 text-amber-600 accent-amber-600"
                  />
                  Historical Mode
                </label>
              </div>
              {historicalMode && genRange && (
                <div className="mt-2 flex items-center gap-3">
                  <span className="text-xs text-gray-500">Gen {genRange.birth_generation}</span>
                  <input
                    type="range"
                    min={genRange.birth_generation}
                    max={genRange.last_generation}
                    value={targetGeneration}
                    onChange={(e) => {
                      setTargetGeneration(Number(e.target.value));
                      setMessages([]);
                    }}
                    className="flex-1 accent-amber-500"
                  />
                  <span className="text-xs text-gray-500">Gen {genRange.last_generation}</span>
                  <span className="rounded bg-amber-900/40 px-2 py-0.5 text-xs font-mono text-amber-300">
                    {targetGeneration}
                  </span>
                </div>
              )}
            </div>
            <div className="flex-1 space-y-3 overflow-y-auto p-4" style={{ maxHeight: '24rem' }}>
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
                      msg.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-800 text-gray-200'
                    }`}
                  >
                    <pre className="whitespace-pre-wrap font-sans">{msg.content}</pre>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="rounded-lg bg-gray-800 px-3 py-2 text-sm text-gray-500">
                    Thinking...
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
            <div className="border-t border-gray-800 p-3 flex gap-2">
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleAsk()}
                placeholder="Ask a question..."
                className="flex-1 rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 placeholder:text-gray-600"
              />
              <button
                onClick={handleAsk}
                disabled={loading || !question.trim()}
                className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
              >
                Ask
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Narratives Tab
// ---------------------------------------------------------------------------

function NarrativesTab({ sessionId, provider, model }: { sessionId: string; provider: string; model: string }) {
  const [generation, setGeneration] = useState(0);
  const [loading, setLoading] = useState(false);
  const [cache, setCache] = useState<Record<number, NarrativeResponse>>({});
  const [error, setError] = useState('');

  const narrative = cache[generation];

  const handleGenerate = async () => {
    if (cache[generation]) return;
    setLoading(true);
    setError('');
    try {
      const resp = await api.getGenerationNarrative(sessionId, generation, provider, model || undefined);
      setCache((prev) => ({ ...prev, [generation]: resp }));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to generate narrative');
    }
    setLoading(false);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-end gap-3">
        <div>
          <label className="mb-1 block text-sm text-gray-400">Generation</label>
          <input
            type="number"
            min={0}
            value={generation}
            onChange={(e) => setGeneration(Number(e.target.value))}
            className="w-28 rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200"
          />
        </div>
        <button
          onClick={handleGenerate}
          disabled={loading}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
        >
          {loading ? 'Generating...' : narrative ? 'Cached' : 'Generate'}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-800 bg-red-900/30 px-4 py-3 text-sm text-red-300">{error}</div>
      )}

      {narrative && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-200">
              Generation {narrative.generation}
            </h3>
            <span className="text-xs text-gray-600">
              {narrative.provider}/{narrative.model} | {narrative.input_tokens + narrative.output_tokens} tokens
            </span>
          </div>
          <div className="prose prose-invert max-w-none text-sm text-gray-300 leading-relaxed">
            <pre className="whitespace-pre-wrap font-sans">{narrative.narrative}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Decision Explorer Tab
// ---------------------------------------------------------------------------

function DecisionExplorerTab({ sessionId, provider, model }: { sessionId: string; provider: string; model: string }) {
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [selectedAgent, setSelectedAgent] = useState('');
  const [agentDetail, setAgentDetail] = useState<AgentDetail | null>(null);
  const [explanations, setExplanations] = useState<Record<number, DecisionExplainResponse>>({});
  const [loadingIdx, setLoadingIdx] = useState<number | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.listAgents(sessionId, { alive_only: false, page_size: 200 })
      .then((res) => setAgents(res.agents))
      .catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    if (!selectedAgent) return;
    setAgentDetail(null);
    setExplanations({});
    api.getAgentDetail(sessionId, selectedAgent).then(setAgentDetail).catch(() => {});
  }, [sessionId, selectedAgent]);

  const decisions = agentDetail?.decision_history ?? [];

  const handleExplain = async (idx: number) => {
    if (explanations[idx]) return;
    setLoadingIdx(idx);
    setError('');
    try {
      const resp = await api.explainDecision(sessionId, selectedAgent, idx, provider, model || undefined);
      setExplanations((prev) => ({ ...prev, [idx]: resp }));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to explain decision');
    }
    setLoadingIdx(null);
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="mb-1 block text-sm text-gray-400">Select Agent</label>
        <select
          className="w-64 rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200"
          value={selectedAgent}
          onChange={(e) => setSelectedAgent(e.target.value)}
        >
          <option value="">Choose an agent...</option>
          {agents.filter((a) => a.has_decisions).map((a) => (
            <option key={a.id} value={a.id}>
              {a.name} ({a.processing_region})
            </option>
          ))}
        </select>
      </div>

      {error && (
        <div className="rounded-lg border border-red-800 bg-red-900/30 px-4 py-3 text-sm text-red-300">{error}</div>
      )}

      {agentDetail && decisions.length === 0 && (
        <div className="text-sm text-gray-500">This agent has no recorded decisions.</div>
      )}

      {decisions.length > 0 && (
        <div className="space-y-3">
          {decisions.map((dec, idx) => {
            const d = dec as Record<string, unknown>;
            const explanation = explanations[idx];
            return (
              <div key={idx} className="rounded-lg border border-gray-800 bg-gray-900 p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-sm font-medium text-gray-200">
                      {(d.context as string) ?? 'Decision'}
                    </span>
                    <span className="ml-2 text-xs text-gray-500">
                      chose: <span className="text-gray-300">{d.chosen_action as string}</span>
                    </span>
                  </div>
                  <button
                    onClick={() => handleExplain(idx)}
                    disabled={loadingIdx === idx || !!explanation}
                    className="rounded-md bg-gray-700 px-3 py-1 text-xs font-medium text-gray-200 hover:bg-gray-600 disabled:opacity-40"
                  >
                    {loadingIdx === idx ? 'Analyzing...' : explanation ? 'Explained' : 'Explain'}
                  </button>
                </div>

                {/* Utility scores */}
                {d.probabilities != null && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {Object.entries(d.probabilities as Record<string, number>).map(([action, prob]: [string, number]) => (
                      <span
                        key={action}
                        className={`rounded px-2 py-0.5 text-xs ${
                          action === d.chosen_action
                            ? 'bg-blue-600/20 text-blue-300'
                            : 'bg-gray-800 text-gray-500'
                        }`}
                      >
                        {action}: {(prob * 100).toFixed(1)}%
                      </span>
                    ))}
                  </div>
                )}

                {explanation && (
                  <div className="mt-3 rounded border border-gray-700 bg-gray-800/50 p-3">
                    <pre className="whitespace-pre-wrap font-sans text-sm text-gray-300 leading-relaxed">
                      {explanation.explanation}
                    </pre>
                    <div className="mt-2 text-xs text-gray-600">
                      {explanation.provider}/{explanation.model} | {explanation.input_tokens + explanation.output_tokens} tokens
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
