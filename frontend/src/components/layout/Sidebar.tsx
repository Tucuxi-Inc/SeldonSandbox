import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Users,
  Flame,
  UserSearch,
  GitCompare,
  GitBranch,
  MapPin,
  Network,
  BookOpen,
  AlertTriangle,
  SlidersHorizontal,
  MessageSquare,
  Crosshair,
  Crown,
  Building2,
  Coins,
  CloudSun,
  Dna,
  BookHeart,
  Brain,
  Map,
  Scale,
  BookUser,
  Newspaper,
} from 'lucide-react';

const coreNavItems = [
  { to: '/', icon: LayoutDashboard, label: 'Mission Control' },
  { to: '/population', icon: Users, label: 'Population' },
  { to: '/suffering', icon: Flame, label: 'Suffering & Contribution' },
  { to: '/agents', icon: UserSearch, label: 'Agent Explorer' },
  { to: '/experiments', icon: GitCompare, label: 'Experiments' },
  { to: '/lineage', icon: GitBranch, label: 'Family & Lineage' },
  { to: '/compare', icon: Scale, label: 'Agent Compare' },
];

const advancedNavItems = [
  { to: '/settlements', icon: MapPin, label: 'Settlements' },
  { to: '/hex-map', icon: Map, label: 'Hex Map' },
  { to: '/network', icon: Network, label: 'Social Network' },
  { to: '/lore', icon: BookOpen, label: 'Lore Evolution' },
  { to: '/anomalies', icon: AlertTriangle, label: 'Anomalies' },
  { to: '/sensitivity', icon: SlidersHorizontal, label: 'Sensitivity' },
  { to: '/outsiders', icon: Crosshair, label: 'Outsiders' },
];

const socialNavItems = [
  { to: '/hierarchy', icon: Crown, label: 'Hierarchy' },
  { to: '/communities', icon: Building2, label: 'Communities' },
];

const economyNavItems = [
  { to: '/economics', icon: Coins, label: 'Economics' },
  { to: '/environment', icon: CloudSun, label: 'Environment' },
];

const scienceNavItems = [
  { to: '/genetics', icon: Dna, label: 'Genetics' },
  { to: '/beliefs', icon: BookHeart, label: 'Beliefs' },
  { to: '/inner-life', icon: Brain, label: 'Inner Life' },
];

const narrativeNavItems = [
  { to: '/interview', icon: MessageSquare, label: 'Agent Interview' },
  { to: '/biography', icon: BookUser, label: 'Biographies' },
  { to: '/chronicle', icon: Newspaper, label: 'Chronicle' },
];

function NavItem({ to, icon: Icon, label }: { to: string; icon: React.ComponentType<{ size?: number }>; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${
          isActive
            ? 'bg-gray-800 text-gray-100'
            : 'text-gray-400 hover:bg-gray-900 hover:text-gray-200'
        }`
      }
    >
      <Icon size={18} />
      {label}
    </NavLink>
  );
}

function SectionDivider({ label }: { label: string }) {
  return (
    <>
      <div className="my-2 border-t border-gray-800" />
      <div className="px-3 py-1 text-xs font-semibold uppercase tracking-wider text-gray-600">
        {label}
      </div>
    </>
  );
}

export function Sidebar() {
  return (
    <aside className="flex w-56 flex-col border-r border-gray-800 bg-gray-950">
      <div className="flex items-center gap-2 border-b border-gray-800 px-4 py-4">
        <div className="h-8 w-8 rounded-lg bg-blue-600 flex items-center justify-center text-sm font-bold">
          S
        </div>
        <div>
          <div className="text-sm font-semibold text-gray-100">Seldon Sandbox</div>
          <div className="text-xs text-gray-500">v0.8.0</div>
        </div>
      </div>
      <nav className="flex-1 space-y-1 overflow-y-auto p-2">
        {coreNavItems.map((item) => (
          <NavItem key={item.to} {...item} />
        ))}

        <SectionDivider label="Advanced" />
        {advancedNavItems.map((item) => (
          <NavItem key={item.to} {...item} />
        ))}

        <SectionDivider label="Social" />
        {socialNavItems.map((item) => (
          <NavItem key={item.to} {...item} />
        ))}

        <SectionDivider label="Economy" />
        {economyNavItems.map((item) => (
          <NavItem key={item.to} {...item} />
        ))}

        <SectionDivider label="Science" />
        {scienceNavItems.map((item) => (
          <NavItem key={item.to} {...item} />
        ))}

        <SectionDivider label="Narrative" />
        {narrativeNavItems.map((item) => (
          <NavItem key={item.to} {...item} />
        ))}
      </nav>
    </aside>
  );
}
