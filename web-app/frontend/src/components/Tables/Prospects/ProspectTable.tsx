import React from 'react';
import { ProspectResponse } from '../../../services/api';
import { Link } from 'react-router-dom';

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
  sortDirection
}) => {
  const renderHeader = () => {
    const commonHeaders = [
      { key: 'name', label: 'Name' },
      { key: 'org', label: 'Team' },
      { key: 'position', label: 'Position' },
      { key: 'age', label: 'Age' },
      { key: 'fv', label: 'FV' },
      { key: 'value', label: 'Trade Value ($M)' },
      { key: 'composite', label: 'Ranking' }
    ];

    const hitterToolHeaders = [
      { key: 'hit', label: 'Hit' },
      { key: 'game', label: 'Game' },
      { key: 'raw', label: 'Raw' },
      { key: 'speed', label: 'Speed' }
    ];

    const pitcherToolHeaders = [
      { key: 'fastball', label: 'Fastball' },
      { key: 'slider', label: 'Slider' },
      { key: 'curve', label: 'Curve' },
      { key: 'change', label: 'Change' },
      { key: 'command', label: 'Command' }
    ];

    return [
      ...commonHeaders,
      ...(playerType === 'hitter' ? hitterToolHeaders : pitcherToolHeaders)
    ];
  };

  const formatValue = (value: number | undefined | null) => {
    if (value === undefined || value === null) return '-';
    return (value / 1000000).toFixed(1);
  };

  const formatRanking = (value: number | undefined | null) => {
    if (value === undefined || value === null) return '-';
    return value.toFixed(2);
  };

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead className="sticky top-0 z-10">
            <tr className="bg-surface-850 border-b border-white/[0.06]">
              {renderHeader().map((header) => (
                <th
                  key={header.key}
                  onClick={() => onSort(header.key)}
                  className="px-3 py-3 text-left text-[11px] font-semibold text-surface-400 uppercase tracking-wider cursor-pointer hover:text-white hover:bg-white/[0.04] select-none transition-colors"
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
              <tr key={i} className={`text-[13px] border-b border-white/[0.03] hover:bg-white/[0.04] transition-colors ${i % 2 === 0 ? 'bg-transparent' : 'bg-white/[0.015]'}`}>
                <td className="px-3 py-2.5 whitespace-nowrap">
                  {player.has_mlb && player.IDfg ? (
                    <Link 
                      to={`/players/${player.IDfg}`} 
                      className="text-accent-blue hover:text-blue-300 font-medium"
                    >
                      {player.name}
                    </Link>
                  ) : (
                    <span className="text-surface-300 font-medium">
                      {player.name}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.org}</td>
                <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.position}</td>
                <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.age}</td>
                <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.fv}</td>
                <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{formatValue(player.value)}</td>
                <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{formatRanking(player.composite)}</td>

                {playerType === 'hitter' ? (
                  <>
                    <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.hit || '-'}</td>
                    <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.game || '-'}</td>
                    <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.raw || '-'}</td>
                    <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.speed || '-'}</td>
                  </>
                ) : (
                  <>
                    <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.fastball || '-'}</td>
                    <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.slider || '-'}</td>
                    <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.curve || '-'}</td>
                    <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.change || '-'}</td>
                    <td className="px-3 py-2.5 whitespace-nowrap text-surface-300 font-mono">{player.command || '-'}</td>
                  </>
                )}
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