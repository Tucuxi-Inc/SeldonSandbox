import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useEffect } from 'react';
import { AppShell } from './components/layout/AppShell';
import { DashboardView } from './components/views/dashboard/DashboardView';
import { PopulationView } from './components/views/population/PopulationView';
import { SufferingView } from './components/views/suffering/SufferingView';
import { AgentExplorerView } from './components/views/agent-explorer/AgentExplorerView';
import { ExperimentView } from './components/views/experiment/ExperimentView';
import { LineageView } from './components/views/lineage/LineageView';
import { SettlementDiagnosticsView } from './components/views/settlements/SettlementDiagnosticsView';
import { NetworkView } from './components/views/network/NetworkView';
import { LoreEvolutionView } from './components/views/lore/LoreEvolutionView';
import { AnomalyDetectionView } from './components/views/anomaly/AnomalyDetectionView';
import { ParameterSensitivityView } from './components/views/sensitivity/ParameterSensitivityView';
import { InterviewView } from './components/views/interview/InterviewView';
import { useSimulationStore } from './stores/simulation';
import { usePolling } from './hooks/useSimulation';

function AppContent() {
  const { refreshSessions } = useSimulationStore();
  usePolling(2000);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<DashboardView />} />
        <Route path="/population" element={<PopulationView />} />
        <Route path="/suffering" element={<SufferingView />} />
        <Route path="/agents" element={<AgentExplorerView />} />
        <Route path="/experiments" element={<ExperimentView />} />
        <Route path="/lineage" element={<LineageView />} />
        <Route path="/settlements" element={<SettlementDiagnosticsView />} />
        <Route path="/network" element={<NetworkView />} />
        <Route path="/lore" element={<LoreEvolutionView />} />
        <Route path="/anomalies" element={<AnomalyDetectionView />} />
        <Route path="/sensitivity" element={<ParameterSensitivityView />} />
        <Route path="/interview" element={<InterviewView />} />
      </Route>
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}
