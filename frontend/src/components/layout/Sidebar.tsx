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
} from 'lucide-react';

const coreNavItems = [
  { to: '/', icon: LayoutDashboard, label: 'Mission Control' },
  { to: '/population', icon: Users, label: 'Population' },
  { to: '/suffering', icon: Flame, label: 'Suffering & Contribution' },
  { to: '/agents', icon: UserSearch, label: 'Agent Explorer' },
  { to: '/experiments', icon: GitCompare, label: 'Experiments' },
  { to: '/lineage', icon: GitBranch, label: 'Family & Lineage' },
];

const advancedNavItems = [
  { to: '/settlements', icon: MapPin, label: 'Settlements' },
  { to: '/network', icon: Network, label: 'Social Network' },
  { to: '/lore', icon: BookOpen, label: 'Lore Evolution' },
  { to: '/anomalies', icon: AlertTriangle, label: 'Anomalies' },
  { to: '/sensitivity', icon: SlidersHorizontal, label: 'Sensitivity' },
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

export function Sidebar() {
  return (
    <aside className="flex w-56 flex-col border-r border-gray-800 bg-gray-950">
      <div className="flex items-center gap-2 border-b border-gray-800 px-4 py-4">
        <div className="h-8 w-8 rounded-lg bg-blue-600 flex items-center justify-center text-sm font-bold">
          S
        </div>
        <div>
          <div className="text-sm font-semibold text-gray-100">Seldon Sandbox</div>
          <div className="text-xs text-gray-500">v0.4.0</div>
        </div>
      </div>
      <nav className="flex-1 space-y-1 overflow-y-auto p-2">
        {coreNavItems.map((item) => (
          <NavItem key={item.to} {...item} />
        ))}

        <div className="my-2 border-t border-gray-800" />
        <div className="px-3 py-1 text-xs font-semibold uppercase tracking-wider text-gray-600">
          Advanced
        </div>

        {advancedNavItems.map((item) => (
          <NavItem key={item.to} {...item} />
        ))}
      </nav>
    </aside>
  );
}
