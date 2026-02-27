import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ProjectionResponse } from '../../../services/api';

interface ProjectionsTableProps {
  data: ProjectionResponse;
  playerType: 'hitter' | 'pitcher';
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  onSort: (key: string) => void;
  sortBy: string;
  sortDirection: 'asc' | 'desc';
}

interface FormattedPlayerRow extends Record<string, any> {
  id: number;
  name: { id: number; name: string };
  team: string;
  position: string;
  age: number;
  status: string;
  base_value: number;
  contract_value: number;
  surplus_value: number;
  // Hitting stats
  g_bat?: number;
  war_bat?: number;
  wrc_plus?: number;
  woba?: number;
  avg?: number;
  obp?: number;
  slg?: number;
  ops?: number;
  bb_pct_bat?: number;
  k_pct_bat?: number;
  hr?: number;
  sb?: number;
  off?: number;
  def_value?: number;
  bsr?: number;
  // Pitching stats
  g_pit?: number;
  gs?: number;
  war_pit?: number;
  era?: number;
  fip?: number;
  k_pct_pit?: number;
  bb_pct_pit?: number;
}
const ProjectionsTable: React.FC<ProjectionsTableProps> = ({ 
  data, 
  playerType,
  currentPage,
  totalPages,
  onPageChange,
  onSort,
  sortBy,
  sortDirection
}) => {

  const hitterHeaders = [
    { key: 'name', label: 'Name' },
    { key: 'team', label: 'Team' },
    { key: 'position', label: 'Pos' },
    { key: 'age', label: 'Age' },
    { key: 'g_bat', label: 'G' },
    { key: 'war_bat', label: 'WAR' },
    { key: 'wrc_plus', label: 'wRC+' },
    { key: 'woba', label: 'wOBA' },
    { key: 'avg', label: 'AVG' },
    { key: 'obp', label: 'OBP' },
    { key: 'slg', label: 'SLG' },
    { key: 'ops', label: 'OPS' },
    { key: 'bb_pct_bat', label: 'BB%' },
    { key: 'k_pct_bat', label: 'K%' },
    { key: 'hr', label: 'HR' },
    { key: 'sb', label: 'SB' },
    { key: 'off', label: 'Off' },
    { key: 'def_value', label: 'Def' },
    { key: 'bsr', label: 'BsR' },
    { key: 'base_value', label: 'Value ($M)' },
    { key: 'contract_value', label: 'Contract ($M)' },
    { key: 'surplus_value', label: 'Surplus ($M)' },
    { key: 'status', label: 'Status' }
  ];

  const pitcherHeaders = [
    { key: 'name', label: 'Name' },
    { key: 'team', label: 'Team' },
    { key: 'position', label: 'Pos' },
    { key: 'age', label: 'Age' },
    { key: 'g_pit', label: 'G' },
    { key: 'gs', label: 'GS' },
    { key: 'war_pit', label: 'WAR' },
    { key: 'era', label: 'ERA' },
    { key: 'fip', label: 'FIP' },
    { key: 'k_pct_pit', label: 'K%' },
    { key: 'bb_pct_pit', label: 'BB%' },
    { key: 'base_value', label: 'Value ($M)' },
    { key: 'contract_value', label: 'Contract ($M)' },
    { key: 'surplus_value', label: 'Surplus ($M)' },
    { key: 'status', label: 'Status' }
  ];

  const headers = playerType === 'hitter' ? hitterHeaders : pitcherHeaders;

  const formatValue = (value: number | undefined | null) => {
    if (value === undefined || value === null) return '-';
    return (value / 1000000).toFixed(1);
  };

  const formatPercent = (value: number | undefined | null) => {
    if (value === undefined || value === null) return '-';
    return `${Math.round(value * 100)}%`;  // Changed to round instead of fixed(1)
  };

  const formatDecimal = (value: number | undefined | null, digits: number = 3) => {
    if (value === undefined || value === null) return '-';
    return value.toFixed(digits);
  };

  const formatCell = (key: string, value: any) => {
    if (value === undefined || value === null) return '-';
    
    if (key === 'name') {
      return (
        <Link to={`/players/${value.id}`} className="text-accent-blue hover:text-blue-300 font-medium">
          {value.name}
        </Link>
      );
    }
  
    switch (key) {
      case 'wrc_plus':
        return formatDecimal(value, 0);
      case 'war_bat':
      case 'war_pit':
      case 'off':
      case 'bsr':
      case 'def_value':
        return formatDecimal(value, 1);
      case 'avg':
      case 'obp':
      case 'slg':
      case 'ops':
      case 'woba':
      case 'era':
      case 'fip':
        return formatDecimal(value, 3);
      case 'k_pct_pit':
      case 'bb_pct_pit':
      case 'bb_pct_bat':
      case 'k_pct_bat':
        return formatPercent(value);
      case 'base_value':
      case 'contract_value':
      case 'surplus_value':
        return formatValue(value);
      default:
        return value.toString();
    }
  };

  const formattedData: FormattedPlayerRow[] = data.players.map(player => ({
    id: player.real_id,
    name: { id: player.mlb_id || player.real_id, name: player.name },
    team: player.team,
    position: player.position,
    age: player.age,
    ...(playerType === 'hitter' ? player.hitting : player.pitching),
    base_value: player.value.base_value,
    contract_value: player.value.contract_value,
    surplus_value: player.value.surplus_value,
    status: player.status,
  }));



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
                  className="px-1.5 py-2 text-left text-[10px] font-semibold text-surface-400 uppercase tracking-wide cursor-pointer hover:text-white hover:bg-white/[0.04] select-none transition-colors"
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
            {formattedData.map((row, i) => (
              <tr key={i} className={`text-[13px] border-b border-white/[0.03] hover:bg-white/[0.04] transition-colors ${i % 2 === 0 ? 'bg-transparent' : 'bg-white/[0.015]'}`}>
                {headers.map((header) => (
                  <td key={header.key} className="px-1.5 py-2 whitespace-nowrap text-[11px] text-surface-300 font-mono">
                    {formatCell(header.key, row[header.key])}
                  </td>
                ))}
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
          <span className="font-medium text-surface-300">{Math.min(currentPage * 50, data.total_count)}</span>
          <span> of </span>
          <span className="font-medium text-surface-300">{data.total_count}</span>
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
}
export default ProjectionsTable;