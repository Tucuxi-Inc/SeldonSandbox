import { useState, useEffect } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type { PresetInfo, ArchetypeInfo } from '../../../types';

export function DashboardView() {
  const { activeSessionId, sessions, refreshAll, setActiveSession } = useSimulationStore();
  const active = sessions.find((s) => s.id === activeSessionId);

  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [archetypes, setArchetypes] = useState<ArchetypeInfo[]>([]);
  const [selectedPreset, setSelectedPreset] = useState('');
  const [config, setConfig] = useState<Record<string, unknown>>({
    initial_population: 100,
    generations_to_run: 50,
    random_seed: 42,
    trait_drift_rate: 0.02,
    lore_enabled: true,
    cognitive_council_enabled: false,
  });
  const [injectionArchetype, setInjectionArchetype] = useState('');
  const [stepping, setStepping] = useState(false);

  useEffect(() => {
    api.getPresets().then(setPresets).catch(() => {});
    api.getArchetypes().then(setArchetypes).catch(() => {});
  }, []);

  const defaults: Record<string, unknown> = {
    initial_population: 100,
    generations_to_run: 50,
    random_seed: 42,
    trait_drift_rate: 0.02,
    lore_enabled: true,
    cognitive_council_enabled: false,
  };

  const handlePresetChange = (name: string) => {
    setSelectedPreset(name);
    const preset = presets.find((p) => p.name === name);
    if (preset) {
      setConfig({ ...defaults, ...preset.config });
    } else {
      setConfig({ ...defaults });
    }
  };

  const handleCreate = async () => {
    const session = await api.createSession({ config, name: config.experiment_name as string });
    setActiveSession(session.id);
    await refreshAll();
  };

  const handleStep = async (n: number) => {
    if (!activeSessionId) return;
    setStepping(true);
    await api.stepSession(activeSessionId, n);
    await refreshAll();
    setStepping(false);
  };

  const handleRunAll = async () => {
    if (!activeSessionId) return;
    setStepping(true);
    await api.runSession(activeSessionId);
    await refreshAll();
    setStepping(false);
  };

  const handleReset = async () => {
    if (!activeSessionId) return;
    await api.resetSession(activeSessionId);
    await refreshAll();
  };

  const handleInject = async () => {
    if (!activeSessionId || !injectionArchetype) return;
    await api.injectOutsider({
      session_id: activeSessionId,
      archetype: injectionArchetype,
    });
    await refreshAll();
  };

  const updateConfig = (key: string, value: unknown) => {
    setConfig({ ...config, [key]: value });
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Mission Control</h1>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Config Panel */}
        <div className="lg:col-span-2 space-y-4">
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-4 text-lg font-semibold text-gray-200">Configuration</h2>

            {/* Preset Selector */}
            <div className="mb-4">
              <label className="mb-1 block text-sm text-gray-400">Preset</label>
              <select
                className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200"
                value={selectedPreset}
                onChange={(e) => handlePresetChange(e.target.value)}
              >
                <option value="">Custom</option>
                {presets.map((p) => (
                  <option key={p.name} value={p.name}>{p.name}</option>
                ))}
              </select>
            </div>

            {/* Sliders */}
            <div className="grid grid-cols-2 gap-4">
              <SliderField label="Population" value={config.initial_population as number} min={10} max={500} step={10} onChange={(v) => updateConfig('initial_population', v)} />
              <SliderField label="Generations" value={config.generations_to_run as number} min={5} max={200} step={5} onChange={(v) => updateConfig('generations_to_run', v)} />
              <SliderField label="Trait Drift Rate" value={config.trait_drift_rate as number} min={0} max={0.5} step={0.01} onChange={(v) => updateConfig('trait_drift_rate', v)} />
              <SliderField label="Random Seed" value={config.random_seed as number} min={0} max={9999} step={1} onChange={(v) => updateConfig('random_seed', v)} />
            </div>

            <div className="mt-4 flex items-center gap-4">
              <ToggleField label="Lore Enabled" value={config.lore_enabled as boolean} onChange={(v) => updateConfig('lore_enabled', v)} />
              <ToggleField label="Cognitive Council" value={config.cognitive_council_enabled as boolean} onChange={(v) => updateConfig('cognitive_council_enabled', v)} />
            </div>
          </div>

          {/* Simulation Controls */}
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-4 text-lg font-semibold text-gray-200">Controls</h2>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleCreate}
                className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
              >
                Create Session
              </button>
              <button
                onClick={() => handleStep(1)}
                disabled={!activeSessionId || stepping || active?.status === 'completed'}
                className="rounded-md bg-gray-700 px-4 py-2 text-sm font-medium text-gray-200 hover:bg-gray-600 disabled:opacity-40"
              >
                Step +1
              </button>
              <button
                onClick={() => handleStep(5)}
                disabled={!activeSessionId || stepping || active?.status === 'completed'}
                className="rounded-md bg-gray-700 px-4 py-2 text-sm font-medium text-gray-200 hover:bg-gray-600 disabled:opacity-40"
              >
                Step +5
              </button>
              <button
                onClick={handleRunAll}
                disabled={!activeSessionId || stepping || active?.status === 'completed'}
                className="rounded-md bg-green-700 px-4 py-2 text-sm font-medium text-white hover:bg-green-600 disabled:opacity-40"
              >
                {stepping ? 'Running...' : 'Run All'}
              </button>
              <button
                onClick={handleReset}
                disabled={!activeSessionId || stepping}
                className="rounded-md bg-gray-700 px-4 py-2 text-sm font-medium text-gray-200 hover:bg-gray-600 disabled:opacity-40"
              >
                Reset
              </button>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="space-y-4">
          {/* Generation Counter */}
          {active && (
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-center">
              <div className="text-sm text-gray-400">Generation</div>
              <div className="text-5xl font-bold text-gray-100">{active.current_generation}</div>
              <div className="text-sm text-gray-500">of {active.max_generations}</div>
              <div className="mt-2 h-2 rounded-full bg-gray-800">
                <div
                  className="h-2 rounded-full bg-blue-600 transition-all"
                  style={{ width: `${(active.current_generation / active.max_generations) * 100}%` }}
                />
              </div>
              <div className="mt-2 text-sm text-gray-400">
                Population: <span className="font-mono text-gray-200">{active.population_size}</span>
              </div>
            </div>
          )}

          {/* Outsider Injection */}
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Inject Outsider</h2>
            <select
              className="mb-2 w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200"
              value={injectionArchetype}
              onChange={(e) => setInjectionArchetype(e.target.value)}
            >
              <option value="">Select archetype...</option>
              {archetypes.map((a) => (
                <option key={a.name} value={a.name}>{a.display_name}</option>
              ))}
            </select>
            <button
              onClick={handleInject}
              disabled={!activeSessionId || !injectionArchetype}
              className="w-full rounded-md bg-amber-700 px-4 py-2 text-sm font-medium text-white hover:bg-amber-600 disabled:opacity-40"
            >
              Inject
            </button>
          </div>

          {/* Archetypes */}
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Archetypes</h2>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {archetypes.map((a) => (
                <div key={a.name} className="rounded border border-gray-800 bg-gray-950 p-2">
                  <div className="text-sm font-medium text-gray-200">{a.display_name}</div>
                  <div className="text-xs text-gray-500">{a.description}</div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {a.key_traits.map((t) => (
                      <span key={t} className="rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-400">{t}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SliderField({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void;
}) {
  const safeValue = value ?? min;
  return (
    <div>
      <div className="flex items-center justify-between">
        <label className="text-sm text-gray-400">{label}</label>
        <span className="font-mono text-sm text-gray-300">{safeValue}</span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step} value={safeValue}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full accent-blue-600"
      />
    </div>
  );
}

function ToggleField({ label, value, onChange }: {
  label: string; value: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
      <input
        type="checkbox"
        checked={value ?? false}
        onChange={(e) => onChange(e.target.checked)}
        className="rounded border-gray-600 bg-gray-800 text-blue-600 accent-blue-600"
      />
      {label}
    </label>
  );
}
