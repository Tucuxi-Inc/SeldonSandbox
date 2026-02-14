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
import { OutsiderTrackerView } from './components/views/outsiders/OutsiderTrackerView';
import { CommunityView } from './components/views/community/CommunityView';
import { EconomicsView } from './components/views/economics/EconomicsView';
import { EnvironmentView } from './components/views/environment/EnvironmentView';
import { HierarchyView } from './components/views/hierarchy/HierarchyView';
import { GeneticsView } from './components/views/genetics/GeneticsView';
import { BeliefsView } from './components/views/beliefs/BeliefsView';
import { InnerLifeView } from './components/views/inner-life/InnerLifeView';
import { HexMapView } from './components/views/hex-map/HexMapView';
import { BiographyView } from './components/views/biography/BiographyView';
import { ChronicleView } from './components/views/chronicle/ChronicleView';
import { AgentComparisonView } from './components/views/comparison/AgentComparisonView';
import { useSimulationStore } from './stores/simulation';
import { usePolling } from './hooks/useSimulation';

function AppContent() {
  const { refreshSessions } = useSimulationStore();
  usePolling();

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
        <Route path="/compare" element={<AgentComparisonView />} />
        <Route path="/settlements" element={<SettlementDiagnosticsView />} />
        <Route path="/hex-map" element={<HexMapView />} />
        <Route path="/network" element={<NetworkView />} />
        <Route path="/lore" element={<LoreEvolutionView />} />
        <Route path="/anomalies" element={<AnomalyDetectionView />} />
        <Route path="/sensitivity" element={<ParameterSensitivityView />} />
        <Route path="/interview" element={<InterviewView />} />
        <Route path="/outsiders" element={<OutsiderTrackerView />} />
        <Route path="/communities" element={<CommunityView />} />
        <Route path="/economics" element={<EconomicsView />} />
        <Route path="/environment" element={<EnvironmentView />} />
        <Route path="/hierarchy" element={<HierarchyView />} />
        <Route path="/genetics" element={<GeneticsView />} />
        <Route path="/beliefs" element={<BeliefsView />} />
        <Route path="/inner-life" element={<InnerLifeView />} />
        <Route path="/biography" element={<BiographyView />} />
        <Route path="/chronicle" element={<ChronicleView />} />
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
