import { useState, useEffect, useRef } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type {
  AgentSummary,
  LLMStatus,
  InterviewResponse,
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

  useEffect(() => {
    api.getLLMStatus().then(setLlmStatus).catch(() => {});
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
}: {
  llmStatus: LLMStatus | null;
  provider: string;
  setProvider: (p: string) => void;
  selectedModel: string;
  setSelectedModel: (m: string) => void;
}) {
  const ollamaModels = llmStatus?.providers.ollama.models ?? [];

  return (
    <div className="max-w-lg space-y-6">
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <h2 className="mb-4 text-lg font-semibold text-gray-200">LLM Provider</h2>

        <div className="space-y-3">
          {/* Anthropic */}
          <label className="flex items-start gap-3 rounded-lg border border-gray-700 p-3 cursor-pointer hover:bg-gray-800/50">
            <input
              type="radio"
              name="provider"
              value="anthropic"
              checked={provider === 'anthropic'}
              onChange={() => { setProvider('anthropic'); setSelectedModel(''); }}
              className="mt-1 accent-blue-600"
            />
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-gray-200">Anthropic (Claude)</span>
                <span className={`rounded px-1.5 py-0.5 text-xs ${
                  llmStatus?.providers.anthropic.available
                    ? 'bg-green-900 text-green-300'
                    : 'bg-red-900 text-red-300'
                }`}>
                  {llmStatus?.providers.anthropic.available ? 'Available' : 'Unavailable'}
                </span>
              </div>
              <p className="mt-1 text-xs text-gray-500">
                Requires <code className="text-gray-400">ANTHROPIC_API_KEY</code> environment variable.
                Uses Claude Sonnet by default.
              </p>
            </div>
          </label>

          {/* Ollama */}
          <label className="flex items-start gap-3 rounded-lg border border-gray-700 p-3 cursor-pointer hover:bg-gray-800/50">
            <input
              type="radio"
              name="provider"
              value="ollama"
              checked={provider === 'ollama'}
              onChange={() => {
                setProvider('ollama');
                if (ollamaModels.length > 0) setSelectedModel(ollamaModels[0]);
              }}
              className="mt-1 accent-blue-600"
            />
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-gray-200">Ollama (Local)</span>
                <span className={`rounded px-1.5 py-0.5 text-xs ${
                  llmStatus?.providers.ollama.available
                    ? 'bg-green-900 text-green-300'
                    : 'bg-red-900 text-red-300'
                }`}>
                  {llmStatus?.providers.ollama.available ? 'Available' : 'Unavailable'}
                </span>
              </div>
              <p className="mt-1 text-xs text-gray-500">
                No API key required. Runs models locally via Ollama. Start Ollama first, then pull models.
              </p>
            </div>
          </label>
        </div>
      </div>

      {/* Model Selection (Ollama) */}
      {provider === 'ollama' && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-3 text-lg font-semibold text-gray-200">Model</h2>
          {ollamaModels.length > 0 ? (
            <select
              className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {ollamaModels.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          ) : (
            <p className="text-sm text-gray-500">
              No models found. Run <code className="text-gray-400">ollama pull llama3.2</code> to download a model.
            </p>
          )}
        </div>
      )}

      {/* Status Summary */}
      {llmStatus && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-3 text-lg font-semibold text-gray-200">Status</h2>
          <p className="text-sm text-gray-400">{llmStatus.message}</p>
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

  useEffect(() => {
    api.listAgents(sessionId, { alive_only: true, page_size: 200 })
      .then((res) => setAgents(res.agents))
      .catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const filteredAgents = agents.filter((a) =>
    a.name.toLowerCase().includes(search.toLowerCase()) ||
    a.id.toLowerCase().includes(search.toLowerCase())
  );

  const handleAsk = async () => {
    if (!selectedAgent || !question.trim()) return;
    const q = question.trim();
    setQuestion('');
    setMessages((prev) => [...prev, { role: 'user', content: q }]);
    setLoading(true);

    try {
      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      const resp = await api.interviewAgent(sessionId, selectedAgent, q, history, provider, model || undefined);
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
              <div className="text-xs text-gray-600">{a.processing_region} | age {a.age}</div>
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
              <span className="text-sm font-medium text-gray-300">
                Interviewing: {agents.find((a) => a.id === selectedAgent)?.name ?? selectedAgent}
              </span>
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
          {agents.map((a) => (
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
                {d.probabilities && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {Object.entries(d.probabilities as Record<string, number>).map(([action, prob]) => (
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
