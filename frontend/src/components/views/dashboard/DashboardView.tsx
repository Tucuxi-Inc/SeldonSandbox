import { useState, useEffect } from 'react';
import { useSimulationStore } from '../../../stores/simulation';
import * as api from '../../../api/client';
import type { PresetInfo, ArchetypeInfo } from '../../../types';
import { WorldPreview } from './WorldPreview';

export function DashboardView() {
  const { activeSessionId, sessions, refreshAll, setActiveSession, updateSession } = useSimulationStore();
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
  // Derive running state from session status
  const isRunning = active?.status === 'running';

  // Outsider builder state
  const [outsiderMode, setOutsiderMode] = useState<'archetype' | 'custom'>('archetype');
  const [injectionArchetype, setInjectionArchetype] = useState('');
  const [outsiderName, setOutsiderName] = useState('');
  const [outsiderGender, setOutsiderGender] = useState('');
  const [outsiderAge, setOutsiderAge] = useState(25);
  const [outsiderNoiseSigma, setOutsiderNoiseSigma] = useState(0.05);
  const [outsiderInjectionGen, setOutsiderInjectionGen] = useState<number | ''>('');
  const [customTraits, setCustomTraits] = useState<Record<string, number>>({});
  const [traitNames, setTraitNames] = useState<string[]>([]);

  // Collapsible config sections
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({});

  useEffect(() => {
    api.getPresets().then(setPresets).catch(() => {});
    api.getArchetypes().then(setArchetypes).catch(() => {});
    api.getTraitNames().then((names) => {
      setTraitNames(names);
      const initial: Record<string, number> = {};
      names.forEach((n) => { initial[n] = 0.5; });
      setCustomTraits(initial);
    }).catch(() => {});
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
    if (!activeSessionId || isRunning) return;
    await api.stepSession(activeSessionId, n);
    await refreshAll();
  };

  const handleRunAll = async () => {
    if (!activeSessionId || isRunning) return;
    const resp = await api.runSession(activeSessionId);
    // Immediately reflect "running" status so polling activates
    updateSession(resp);
    // Then do a full refresh to pick up metrics/agents
    refreshAll();
  };

  const handleReset = async () => {
    if (!activeSessionId) return;
    await api.resetSession(activeSessionId);
    await refreshAll();
  };

  const handleFork = async () => {
    if (!activeSessionId) return;
    try {
      const cloned = await api.cloneSession(activeSessionId, {
        name: `${active?.name ?? 'Session'} (Fork)`,
      });
      setActiveSession(cloned.id);
      await refreshAll();
    } catch {
      // ignore
    }
  };

  const handleInject = async () => {
    if (!activeSessionId) return;
    const gen = outsiderInjectionGen === '' ? undefined : outsiderInjectionGen;
    if (outsiderMode === 'archetype') {
      if (!injectionArchetype) return;
      await api.injectOutsider({
        session_id: activeSessionId,
        archetype: injectionArchetype,
        noise_sigma: outsiderNoiseSigma,
        name: outsiderName || undefined,
        injection_generation: gen,
      });
    } else {
      await api.injectOutsider({
        session_id: activeSessionId,
        custom_traits: customTraits,
        name: outsiderName || undefined,
        gender: outsiderGender || undefined,
        age: outsiderAge,
        injection_generation: gen,
      });
    }
    await refreshAll();
  };

  const updateConfig = (key: string, value: unknown) => {
    setConfig({ ...config, [key]: value });
  };

  const updateNestedConfig = (section: string, key: string, value: unknown) => {
    const current = (config[section] as Record<string, unknown>) ?? {};
    setConfig({ ...config, [section]: { ...current, [key]: value } });
  };

  const toggleExtension = (name: string, enabled: boolean, deps?: string[], dependents?: string[]) => {
    const current = ((config.extensions_enabled as string[]) ?? []).slice();
    if (enabled) {
      if (!current.includes(name)) current.push(name);
      // Auto-enable dependencies
      deps?.forEach((d) => { if (!current.includes(d)) current.push(d); });
    } else {
      const idx = current.indexOf(name);
      if (idx >= 0) current.splice(idx, 1);
      // Auto-disable dependents
      dependents?.forEach((d) => {
        const dIdx = current.indexOf(d);
        if (dIdx >= 0) current.splice(dIdx, 1);
      });
    }
    updateConfig('extensions_enabled', current);
  };

  const isExtEnabled = (name: string) => ((config.extensions_enabled as string[]) ?? []).includes(name);

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
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
              <SliderField label="Random Seed" value={config.random_seed as number} min={1} max={9999} step={1} onChange={(v) => updateConfig('random_seed', v)} />
            </div>

            <div className="mt-4 flex items-center gap-4">
              <ToggleField label="Lore Enabled" value={config.lore_enabled as boolean} onChange={(v) => updateConfig('lore_enabled', v)} />
              <ToggleField label="Cognitive Council" value={config.cognitive_council_enabled as boolean} onChange={(v) => updateConfig('cognitive_council_enabled', v)} />
              <ToggleField
                label="Tick Engine"
                value={!!((config.tick_config as Record<string, unknown>)?.enabled)}
                onChange={(v) => updateNestedConfig('tick_config', 'enabled', v)}
              />
              <ToggleField
                label="Hex Grid"
                value={!!((config.hex_grid_config as Record<string, unknown>)?.enabled)}
                onChange={(v) => {
                  setConfig((prev) => {
                    const hexCfg = (prev.hex_grid_config as Record<string, unknown>) ?? {};
                    const tickCfg = (prev.tick_config as Record<string, unknown>) ?? {};
                    const next: Record<string, unknown> = { ...prev, hex_grid_config: { ...hexCfg, enabled: v } };
                    if (v) next.tick_config = { ...tickCfg, enabled: true };
                    return next;
                  });
                }}
              />
            </div>

            {/* Extension Toggles */}
            <div className="mt-4">
              <div className="mb-2 text-sm font-medium text-gray-400">Extensions</div>
              <div className="flex flex-wrap items-center gap-4">
                <ToggleField
                  label="Geography"
                  value={isExtEnabled('geography')}
                  onChange={(v) => toggleExtension('geography', v, [], ['migration', 'diplomacy', 'environment'])}
                />
                <ToggleField
                  label="Migration"
                  value={isExtEnabled('migration')}
                  onChange={(v) => toggleExtension('migration', v, ['geography'])}
                />
                <ToggleField
                  label="Resources"
                  value={isExtEnabled('resources')}
                  onChange={(v) => toggleExtension('resources', v)}
                />
                <ToggleField
                  label="Technology"
                  value={isExtEnabled('technology')}
                  onChange={(v) => toggleExtension('technology', v)}
                />
                <ToggleField
                  label="Culture"
                  value={isExtEnabled('culture')}
                  onChange={(v) => toggleExtension('culture', v)}
                />
                <ToggleField
                  label="Conflict"
                  value={isExtEnabled('conflict')}
                  onChange={(v) => toggleExtension('conflict', v)}
                />
                <ToggleField
                  label="Social Dynamics"
                  value={isExtEnabled('social_dynamics')}
                  onChange={(v) => toggleExtension('social_dynamics', v)}
                />
                <ToggleField
                  label="Diplomacy"
                  value={isExtEnabled('diplomacy')}
                  onChange={(v) => toggleExtension('diplomacy', v, ['geography'])}
                />
                <ToggleField
                  label="Economics"
                  value={isExtEnabled('economics')}
                  onChange={(v) => toggleExtension('economics', v)}
                />
                <ToggleField
                  label="Environment"
                  value={isExtEnabled('environment')}
                  onChange={(v) => toggleExtension('environment', v, ['geography'])}
                />
                <ToggleField
                  label="Epistemology"
                  value={isExtEnabled('epistemology')}
                  onChange={(v) => toggleExtension('epistemology', v)}
                />
                <ToggleField
                  label="Inner Life"
                  value={isExtEnabled('inner_life')}
                  onChange={(v) => toggleExtension('inner_life', v)}
                />
              </div>
            </div>
          </div>

          {/* Collapsible Config Sections */}
          <CollapsibleSection title="Genetics & Epigenetics" expanded={!!expandedSections['genetics']} onToggle={() => toggleSection('genetics')}>
            <div className="grid grid-cols-2 gap-4">
              <div className="col-span-2 flex items-center gap-4">
                <ToggleField label="Genetics Enabled" value={!!((config.genetics_config as Record<string, unknown>)?.genetics_enabled)} onChange={(v) => updateNestedConfig('genetics_config', 'genetics_enabled', v)} />
                <ToggleField label="Epigenetics Enabled" value={!!((config.epigenetics_config as Record<string, unknown>)?.epigenetics_enabled)} onChange={(v) => updateNestedConfig('epigenetics_config', 'epigenetics_enabled', v)} />
              </div>
              <SliderField label="Mutation Rate" value={((config.genetics_config as Record<string, unknown>)?.mutation_rate as number) ?? 0.001} min={0} max={0.01} step={0.001} onChange={(v) => updateNestedConfig('genetics_config', 'mutation_rate', v)} />
              <SliderField label="Gene-Trait Influence" value={((config.genetics_config as Record<string, unknown>)?.gene_trait_influence as number) ?? 0.3} min={0} max={1} step={0.05} onChange={(v) => updateNestedConfig('genetics_config', 'gene_trait_influence', v)} />
              <SliderField label="Transgenerational Rate" value={((config.epigenetics_config as Record<string, unknown>)?.transgenerational_rate as number) ?? 0.3} min={0} max={1} step={0.05} onChange={(v) => updateNestedConfig('epigenetics_config', 'transgenerational_rate', v)} />
            </div>
          </CollapsibleSection>

          {isExtEnabled('social_dynamics') && (
            <CollapsibleSection title="Social Dynamics" expanded={!!expandedSections['social']} onToggle={() => toggleSection('social')}>
              <div className="grid grid-cols-2 gap-4">
                <SliderField label="Status Contribution Weight" value={((config.hierarchy_config as Record<string, unknown>)?.contribution_weight as number) ?? 0.4} min={0} max={1} step={0.05} onChange={(v) => updateNestedConfig('hierarchy_config', 'contribution_weight', v)} />
                <SliderField label="Status Age Weight" value={((config.hierarchy_config as Record<string, unknown>)?.age_weight as number) ?? 0.2} min={0} max={1} step={0.05} onChange={(v) => updateNestedConfig('hierarchy_config', 'age_weight', v)} />
                <SliderField label="Influence Decay Rate" value={((config.hierarchy_config as Record<string, unknown>)?.influence_decay as number) ?? 0.1} min={0} max={0.5} step={0.05} onChange={(v) => updateNestedConfig('hierarchy_config', 'influence_decay', v)} />
                <div className="col-span-2 flex items-center gap-4">
                  <ToggleField label="Mentorship Enabled" value={((config.mentorship_config as Record<string, unknown>)?.enabled as boolean) ?? true} onChange={(v) => updateNestedConfig('mentorship_config', 'enabled', v)} />
                </div>
                <SliderField label="Max Mentees" value={((config.mentorship_config as Record<string, unknown>)?.max_mentees as number) ?? 3} min={1} max={10} step={1} onChange={(v) => updateNestedConfig('mentorship_config', 'max_mentees', v)} />
              </div>
            </CollapsibleSection>
          )}

          {isExtEnabled('diplomacy') && (
            <CollapsibleSection title="Diplomacy" expanded={!!expandedSections['diplomacy']} onToggle={() => toggleSection('diplomacy')}>
              <div className="grid grid-cols-2 gap-4">
                <SliderField label="Alliance Threshold" value={((config.diplomacy_config as Record<string, unknown>)?.alliance_threshold as number) ?? 0.6} min={0} max={1} step={0.05} onChange={(v) => updateNestedConfig('diplomacy_config', 'alliance_threshold', v)} />
                <SliderField label="Rivalry Threshold" value={((config.diplomacy_config as Record<string, unknown>)?.rivalry_threshold as number) ?? -0.4} min={-1} max={0} step={0.05} onChange={(v) => updateNestedConfig('diplomacy_config', 'rivalry_threshold', v)} />
                <SliderField label="Cultural Exchange Rate" value={((config.diplomacy_config as Record<string, unknown>)?.cultural_exchange_rate as number) ?? 0.1} min={0} max={0.5} step={0.05} onChange={(v) => updateNestedConfig('diplomacy_config', 'cultural_exchange_rate', v)} />
              </div>
            </CollapsibleSection>
          )}

          {isExtEnabled('economics') && (
            <CollapsibleSection title="Economics" expanded={!!expandedSections['economics']} onToggle={() => toggleSection('economics')}>
              <div className="grid grid-cols-2 gap-4">
                <SliderField label="Base Production Rate" value={((config.economics_config as Record<string, unknown>)?.base_production_rate as number) ?? 1.0} min={0.1} max={5} step={0.1} onChange={(v) => updateNestedConfig('economics_config', 'base_production_rate', v)} />
                <SliderField label="Trade Distance Cost" value={((config.economics_config as Record<string, unknown>)?.trade_distance_cost as number) ?? 0.1} min={0} max={0.5} step={0.05} onChange={(v) => updateNestedConfig('economics_config', 'trade_distance_cost', v)} />
                <SliderField label="Poverty Threshold" value={((config.economics_config as Record<string, unknown>)?.poverty_threshold as number) ?? 0.2} min={0} max={1} step={0.05} onChange={(v) => updateNestedConfig('economics_config', 'poverty_threshold', v)} />
              </div>
            </CollapsibleSection>
          )}

          {isExtEnabled('environment') && (
            <CollapsibleSection title="Environment" expanded={!!expandedSections['environment']} onToggle={() => toggleSection('environment')}>
              <div className="grid grid-cols-2 gap-4">
                <div className="col-span-2">
                  <ToggleField label="Seasons Enabled" value={((config.environment_config as Record<string, unknown>)?.seasons_enabled as boolean) ?? true} onChange={(v) => updateNestedConfig('environment_config', 'seasons_enabled', v)} />
                </div>
                <SliderField label="Season Length" value={((config.environment_config as Record<string, unknown>)?.season_length as number) ?? 5} min={1} max={20} step={1} onChange={(v) => updateNestedConfig('environment_config', 'season_length', v)} />
                <SliderField label="Drought Probability" value={((config.environment_config as Record<string, unknown>)?.drought_probability as number) ?? 0.05} min={0} max={0.5} step={0.01} onChange={(v) => updateNestedConfig('environment_config', 'drought_probability', v)} />
                <SliderField label="Plague Probability" value={((config.environment_config as Record<string, unknown>)?.plague_probability as number) ?? 0.02} min={0} max={0.5} step={0.01} onChange={(v) => updateNestedConfig('environment_config', 'plague_probability', v)} />
                <SliderField label="Climate Drift Rate" value={((config.environment_config as Record<string, unknown>)?.climate_drift_rate as number) ?? 0.001} min={0} max={0.01} step={0.001} onChange={(v) => updateNestedConfig('environment_config', 'climate_drift_rate', v)} />
              </div>
            </CollapsibleSection>
          )}

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
                disabled={!activeSessionId || isRunning || active?.status === 'completed'}
                className="rounded-md bg-gray-700 px-4 py-2 text-sm font-medium text-gray-200 hover:bg-gray-600 disabled:opacity-40"
              >
                Step +1
              </button>
              <button
                onClick={() => handleStep(5)}
                disabled={!activeSessionId || isRunning || active?.status === 'completed'}
                className="rounded-md bg-gray-700 px-4 py-2 text-sm font-medium text-gray-200 hover:bg-gray-600 disabled:opacity-40"
              >
                Step +5
              </button>
              <button
                onClick={handleRunAll}
                disabled={!activeSessionId || isRunning || active?.status === 'completed'}
                className="rounded-md bg-green-700 px-4 py-2 text-sm font-medium text-white hover:bg-green-600 disabled:opacity-40"
              >
                {isRunning ? 'Running...' : 'Run All'}
              </button>
              <button
                onClick={handleReset}
                disabled={!activeSessionId || isRunning}
                className="rounded-md bg-gray-700 px-4 py-2 text-sm font-medium text-gray-200 hover:bg-gray-600 disabled:opacity-40"
              >
                Reset
              </button>
              <button
                onClick={handleFork}
                disabled={!activeSessionId || isRunning || active?.current_generation === 0}
                className="rounded-md bg-purple-700 px-4 py-2 text-sm font-medium text-white hover:bg-purple-600 disabled:opacity-40"
                title="Fork this session to create a what-if branch"
              >
                Fork
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
              {isRunning && (
                <div className="mt-2 text-sm text-blue-400 animate-pulse">
                  Simulating generation {active.current_generation} of {active.max_generations}...
                </div>
              )}
              {active.status === 'error' && (
                <div className="mt-2 text-sm text-red-400">
                  Simulation error — check server logs
                </div>
              )}
            </div>
          )}

          {/* Outsider Builder */}
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <h2 className="mb-3 text-lg font-semibold text-gray-200">Outsider Builder</h2>

            {/* Mode Tabs */}
            <div className="mb-3 flex rounded-md border border-gray-700 overflow-hidden">
              <button
                className={`flex-1 py-1.5 text-xs font-medium ${outsiderMode === 'archetype' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}
                onClick={() => setOutsiderMode('archetype')}
              >
                Archetype
              </button>
              <button
                className={`flex-1 py-1.5 text-xs font-medium ${outsiderMode === 'custom' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}
                onClick={() => setOutsiderMode('custom')}
              >
                Custom Build
              </button>
            </div>

            {/* Common fields */}
            <div className="mb-2">
              <label className="mb-1 block text-xs text-gray-500">Name (optional)</label>
              <input
                type="text"
                className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-gray-200"
                value={outsiderName}
                onChange={(e) => setOutsiderName(e.target.value)}
                placeholder="Auto-generated if empty"
              />
            </div>
            <div className="mb-2">
              <label className="mb-1 block text-xs text-gray-500">Injection Generation</label>
              <input
                type="number"
                className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-gray-200"
                value={outsiderInjectionGen}
                onChange={(e) => setOutsiderInjectionGen(e.target.value === '' ? '' : Number(e.target.value))}
                placeholder={`Current (${active?.current_generation ?? 0})`}
              />
            </div>

            {outsiderMode === 'archetype' ? (
              <>
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
                <SliderField label="Noise Sigma" value={outsiderNoiseSigma} min={0} max={0.3} step={0.01} onChange={setOutsiderNoiseSigma} />
              </>
            ) : (
              <>
                <div className="mb-2">
                  <label className="mb-1 block text-xs text-gray-500">Gender</label>
                  <select
                    className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-gray-200"
                    value={outsiderGender}
                    onChange={(e) => setOutsiderGender(e.target.value)}
                  >
                    <option value="">Unspecified</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="non-binary">Non-binary</option>
                  </select>
                </div>
                <SliderField label="Age" value={outsiderAge} min={15} max={60} step={1} onChange={setOutsiderAge} />
                <div className="mt-2 max-h-48 overflow-y-auto space-y-1">
                  <div className="text-xs text-gray-500 mb-1">Trait Sliders</div>
                  {traitNames.map((name) => (
                    <MiniSlider
                      key={name}
                      label={name}
                      value={customTraits[name] ?? 0.5}
                      onChange={(v) => setCustomTraits({ ...customTraits, [name]: v })}
                    />
                  ))}
                </div>
              </>
            )}

            <button
              onClick={handleInject}
              disabled={!activeSessionId || (outsiderMode === 'archetype' && !injectionArchetype)}
              className="mt-3 w-full rounded-md bg-amber-700 px-4 py-2 text-sm font-medium text-white hover:bg-amber-600 disabled:opacity-40"
            >
              Inject Outsider
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

      {/* World Preview — shown when active session has tick/hex enabled */}
      <WorldPreview />
    </div>
  );
}

function CollapsibleSection({ title, expanded, onToggle, children }: {
  title: string; expanded: boolean; onToggle: () => void; children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900">
      <button
        onClick={onToggle}
        className="flex w-full items-center justify-between p-4 text-left"
      >
        <h2 className="text-sm font-semibold text-gray-200">{title}</h2>
        <span className="text-gray-500 text-xs">{expanded ? 'Collapse' : 'Expand'}</span>
      </button>
      {expanded && <div className="border-t border-gray-800 p-4">{children}</div>}
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

function MiniSlider({ label, value, onChange }: {
  label: string; value: number; onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-28 truncate text-xs text-gray-500">{label}</span>
      <input
        type="range"
        min={0} max={1} step={0.05} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 accent-blue-600"
      />
      <span className="w-8 text-right font-mono text-xs text-gray-400">{value.toFixed(2)}</span>
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
