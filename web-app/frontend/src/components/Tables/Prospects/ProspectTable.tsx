import React from 'react';
import { ProspectResponse } from '../../../services/api';
import { Link } from 'react-router-dom';

type ViewMode = 'grades' | 'stats' | 'all_stats';

interface ProspectsTableProps {
  data: ProspectResponse;
  playerType: 'hitter' | 'pitcher';
  year: number;
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  onSort: (key: string) => void;
  sortBy: string;
  sortDirection: 'asc' | 'desc';
  viewMode: ViewMode;
}

const ProspectsTable: React.FC<ProspectsTableProps> = ({ 
  data, 
  playerType, 
  year,
  currentPage,
  totalPages,
  onPageChange,
  onSort,
  sortBy,
  sortDirection,
  viewMode,
}) => {
  // ── Column definitions per view mode ──────────────────────────────────
  const coreHeaders = [
    { key: 'name', label: 'Name' },
    { key: 'org', label: 'Team' },
    { key: 'position', label: 'Pos' },
    { key: 'age', label: 'Age' },
    { key: 'fv', label: 'FV' },
    { key: 'top_100', label: 'T100' },
    { key: 'org_rank', label: 'Org#' },
    { key: 'value', label: 'Value ($M)' },
  ];

  const hitterGradeHeaders = [
    { key: 'hit', label: 'Hit' },
    { key: 'game', label: 'Game' },
    { key: 'raw', label: 'Raw' },
    { key: 'speed', label: 'Spd' },
  ];

  const pitcherGradeHeaders = [
    { key: 'fastball', label: 'FB' },
    { key: 'slider', label: 'SL' },
    { key: 'curve', label: 'CB' },
    { key: 'change', label: 'CH' },
    { key: 'command', label: 'CMD' },
  ];

  const hitterStatHeaders = [
    { key: 'wrc_plus', label: 'wRC+' },
    { key: 'avg', label: 'AVG' },
    { key: 'obp', label: 'OBP' },
    { key: 'slg', label: 'SLG' },
    { key: 'iso', label: 'ISO' },
    { key: 'bb_pct', label: 'BB%' },
    { key: 'k_pct', label: 'K%' },
    { key: 'pa', label: 'PA' },
    { key: 'babip', label: 'BABIP' },
    { key: 'woba', label: 'wOBA' },
    { key: 'spd_stat', label: 'Spd' },
  ];

  const pitcherStatHeaders = [
    { key: 'era', label: 'ERA' },
    { key: 'fip', label: 'FIP' },
    { key: 'xfip', label: 'xFIP' },
    { key: 'k_9', label: 'K/9' },
    { key: 'bb_9', label: 'BB/9' },
    { key: 'k_pct', label: 'K%' },
    { key: 'bb_pct', label: 'BB%' },
    { key: 'whip', label: 'WHIP' },
    { key: 'ip', label: 'IP' },
    { key: 'babip', label: 'BABIP' },
  ];

  const getHeaders = () => {
    const gradeHeaders = playerType === 'hitter' ? hitterGradeHeaders : pitcherGradeHeaders;
    const statHeaders = playerType === 'hitter' ? hitterStatHeaders : pitcherStatHeaders;

    switch (viewMode) {
      case 'grades':
        return [...coreHeaders, ...gradeHeaders];
      case 'stats':
        return [...coreHeaders, ...gradeHeaders, ...statHeaders];
      case 'all_stats':
        return [...coreHeaders, ...statHeaders];
    }
  };

  const formatValue = (value: number | undefined | null) => {
    if (value === undefined || value === null) return '-';
    return (value / 1000000).toFixed(1);
  };

  const formatStat = (key: string, value: number | string | undefined | null): string => {
    if (value === undefined || value === null) return '-';
    const v = typeof value === 'string' ? parseFloat(value) : value;
    if (isNaN(v)) return String(value);
    if (key === 'pa' || key === 'g') return v.toFixed(0);
    if (key === 'wrc_plus') return v.toFixed(0);
    if (key === 'ip') return v.toFixed(1);
    if (key === 'avg' || key === 'obp' || key === 'slg' || key === 'iso' || key === 'babip' || key === 'woba') return v.toFixed(3);
    if (key === 'bb_pct' || key === 'k_pct') return v.toFixed(1) + '%';
    if (key === 'era' || key === 'fip' || key === 'xfip' || key === 'whip') return v.toFixed(2);
    if (key === 'k_9' || key === 'bb_9' || key === 'k_bb' || key === 'hr_9') return v.toFixed(2);
    if (key === 'spd_stat') return v.toFixed(1);
    return v.toFixed(1);
  };

  const getStatValue = (player: typeof data.players[0], key: string): number | string | undefined | null => {
    const stats = player.latest_stats;
    if (!stats) return null;
    // Map table key → stats object key
    const statKey = key === 'spd_stat' ? 'spd' : key;
    return (stats as Record<string, unknown>)[statKey] as number | string | undefined | null;
  };

  const getCellValue = (player: typeof data.players[0], key: string): string => {
    switch (key) {
      case 'name': return player.name;
      case 'org': return player.org;
      case 'position': return player.position;
      case 'age': return player.age ? Math.floor(player.age).toString() : '-';
      case 'fv': return player.fv || '-';
      case 'top_100': return player.top_100 ? '#' + player.top_100 : '-';
      case 'org_rank': return player.org_rank ? '#' + player.org_rank : '-';
      case 'value': return formatValue(player.value);
      // Grades
      case 'hit': case 'game': case 'raw': case 'speed':
      case 'fastball': case 'slider': case 'curve': case 'change': case 'command':
        return (player as Record<string, unknown>)[key] as string || '-';
      // Stats
      default:
        return formatStat(key, getStatValue(player, key));
    }
  };

  // No grade color coding — keep all grade values uniform
  const gradeColor = (_grade: string | null | undefined) => '';

  // No stat color coding — keep all stat values uniform
  const statColor = (_key: string, _value: number | string | undefined | null) => '';

  const isGradeColumn = (key: string) =>
    ['hit', 'game', 'raw', 'speed', 'fastball', 'slider', 'curve', 'change', 'command'].includes(key);

  const isStatColumn = (key: string) =>
    !isGradeColumn(key) && ![
      'name', 'org', 'position', 'age', 'fv', 'top_100', 'org_rank', 'value'
    ].includes(key);

  const headers = getHeaders();

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead className="sticky top-0 z-10">
            <tr className="bg-surface-850 border-b border-white/[0.06]">
              {headers.map((header) => (
                <th
                  key={header.key}
                  onClick={() => onSort(header.key)}
                  className="px-1.5 py-2 text-left text-[10px] font-semibold text-surface-400 uppercase tracking-wide cursor-pointer hover:text-white hover:bg-white/[0.04] select-none transition-colors whitespace-nowrap"
                >
                  <div className="flex items-center gap-1">
                    <span>{header.label}</span>
                    {sortBy === header.key && (
                      <span className="text-brand-400">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.players.map((player, i) => (
              <tr key={player.id ?? i} className={`text-[11px] border-b border-white/[0.03] hover:bg-white/[0.04] transition-colors ${i % 2 === 0 ? 'bg-transparent' : 'bg-white/[0.015]'}`}>
                {headers.map((header) => {
                  if (header.key === 'name') {
                    return (
                      <td key={header.key} className="px-1.5 py-2 whitespace-nowrap">
                        {player.has_mlb && player.IDfg ? (
                          <Link
                            to={`/players/${player.IDfg}`}
                            className="text-accent-blue hover:text-blue-300 font-medium"
                          >
                            {player.name}
                          </Link>
                        ) : player.id ? (
                          <Link
                            to={`/prospects/${player.id}`}
                            className="text-blue-400 hover:text-blue-300 font-medium transition-colors"
                          >
                            {player.name}
                          </Link>
                        ) : (
                          <span className="text-surface-300 font-medium">{player.name}</span>
                        )}
                      </td>
                    );
                  }
                  if (header.key === 'fv') {
                    return (
                      <td key={header.key} className="px-1.5 py-2 whitespace-nowrap font-mono text-surface-300">
                        {player.fv || '-'}
                      </td>
                    );
                  }

                  const rawValue = getCellValue(player, header.key);
                  let colorClass = '';
                  if (isGradeColumn(header.key)) {
                    colorClass = gradeColor(rawValue);
                  } else if (isStatColumn(header.key)) {
                    colorClass = statColor(header.key, getStatValue(player, header.key));
                  }

                  return (
                    <td
                      key={header.key}
                      className={`px-1.5 py-2 whitespace-nowrap font-mono text-surface-300 ${colorClass}`}
                    >
                      {rawValue}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination Controls */}
      <div className="flex items-center justify-between px-4 py-3 border-t border-white/[0.06]">
        <p className="text-sm text-surface-500">
          <span className="font-medium text-surface-300">{(currentPage - 1) * 50 + 1}</span>
          <span>–</span>
          <span className="font-medium text-surface-300">{Math.min(currentPage * 50, data.count)}</span>
          <span> of </span>
          <span className="font-medium text-surface-300">{data.count}</span>
        </p>
        
        <div className="flex items-center gap-2">
          <button
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage === 1}
            className="px-3 py-1.5 rounded-lg text-sm font-medium text-surface-300 
                    bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06]
                    disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            ← Prev
          </button>
          
          <span className="text-sm text-surface-500 tabular-nums">
            {currentPage} / {totalPages}
          </span>
          
          <button
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="px-3 py-1.5 rounded-lg text-sm font-medium text-surface-300 
                    bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06]
                    disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
};

export default ProspectsTable;