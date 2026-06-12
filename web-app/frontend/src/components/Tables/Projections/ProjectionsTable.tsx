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
  bat?: number;
  def_value?: number;
  bsr?: number;
  // Pitching stats
  g_pit?: number;
  gs?: number;
  ip?: number;
  war_pit?: number;
  era?: number;
  fip?: number;
  k_pct_pit?: number;
  bb_pct_pit?: number;
  gb_pct?: number;
  fb_pct?: number;
  hr_fb?: number;
  hr_9?: number;
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
    { key: 'bat', label: 'Bat' },
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
    { key: 'ip', label: 'IP' },
    { key: 'war_pit', label: 'WAR' },
    { key: 'era', label: 'ERA' },
    { key: 'fip', label: 'FIP' },
    { key: 'k_pct_pit', label: 'K%' },
    { key: 'bb_pct_pit', label: 'BB%' },
    { key: 'gb_pct', label: 'GB%' },
    { key: 'fb_pct', label: 'FB%' },
    { key: 'hr_9', label: 'HR/9' },
    { key: 'hr_fb', label: 'HR/FB' },
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
      case 'bat':
      case 'bsr':
      case 'def_value':
        return formatDecimal(value, 1);
      case 'avg':
      case 'obp':
      case 'slg':
      case 'ops':
      case 'woba':
        return formatDecimal(value, 3);
      case 'ip':
        return formatDecimal(value, 1);
      case 'era':
      case 'fip':
      case 'hr_9':
        return formatDecimal(value, 2);
      case 'gb_pct':
      case 'fb_pct':
      case 'hr_fb':
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



  const mobileHeaders = playerType === 'hitter'
    ? [
        { key: 'name', label: 'Name' },
        { key: 'position', label: 'Pos' },
        { key: 'war_bat', label: 'WAR' },
        { key: 'base_value', label: 'Value ($M)' }
      ]
    : [
        { key: 'name', label: 'Name' },
        { key: 'position', label: 'Pos' },
        { key: 'war_pit', label: 'WAR' },
        { key: 'base_value', label: 'Value ($M)' }
      ];

  return (
    <div>
      {/* Mobile Card View */}
      <div className="md:hidden space-y-3">
        {formattedData.map((row, i) => (
          <div
            key={i}
            className="p-4 rounded-lg border border-gray-200 bg-white hover:bg-gray-50 transition-colors"
          >
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="font-semibold text-gray-900 text-sm">{row.name.name}</h3>
                <p className="text-xs text-gray-500 mt-0.5">{row.team} • {row.position} • Age {row.age}</p>
              </div>
              <span className="text-xs font-semibold text-brand-500 bg-brand-500/10 px-2.5 py-1 rounded whitespace-nowrap">
                ${formatValue(row.base_value)}M
              </span>
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <p className="text-gray-500">{playerType === 'hitter' ? 'WAR' : 'WAR'}</p>
                <p className="font-semibold text-gray-900">{formatCell(playerType === 'hitter' ? 'war_bat' : 'war_pit', row[playerType === 'hitter' ? 'war_bat' : 'war_pit'])}</p>
              </div>
              <div>
                <p className="text-gray-500">{playerType === 'hitter' ? 'wRC+' : 'ERA'}</p>
                <p className="font-semibold text-gray-900">{playerType === 'hitter' ? formatCell('wrc_plus', row.wrc_plus) : formatCell('era', row.era)}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Desktop Table View */}
      <div className="hidden md:block overflow-x-auto">
        <table className="min-w-full">
          <thead className="sticky top-0 z-10">
            <tr className="bg-gray-50 border-b border-gray-200">
              {headers.map((header) => (
                <th
                  key={header.key}
                  onClick={() => onSort(header.key)}
                  className="px-2 py-1.5 text-left text-[10px] font-semibold text-gray-500 uppercase tracking-wide cursor-pointer hover:text-gray-900 hover:bg-gray-100 select-none transition-colors"
                >
                  <div className="flex items-center gap-1">
                    <span>{header.label}</span>
                    {sortBy === header.key && (
                      <span className="text-brand-500">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {formattedData.map((row, i) => (
              <tr key={i} className={`border-b border-gray-100 hover:bg-gray-50 transition-colors ${i % 2 === 0 ? 'bg-transparent' : 'bg-gray-50/50'}`}>
                {headers.map((header) => (
                  <td key={header.key} className="px-2 py-1.5 whitespace-nowrap text-xs text-gray-600">
                    {formatCell(header.key, row[header.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination Controls */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 px-4 py-4 border-t border-gray-200">
        <p className="text-xs sm:text-sm text-gray-500 order-2 sm:order-1">
          <span className="font-medium text-gray-600">{(currentPage - 1) * 50 + 1}</span>
          <span>–</span>
          <span className="font-medium text-gray-600">{Math.min(currentPage * 50, data.total_count)}</span>
          <span> of </span>
          <span className="font-medium text-gray-600">{data.total_count}</span>
        </p>
        
        <div className="flex items-center gap-2 order-1 sm:order-2">
          <button
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage === 1}
            className="min-h-10 min-w-10 px-3 py-2 rounded text-xs sm:text-sm font-medium text-gray-600 
                    bg-gray-50 hover:bg-gray-100 border border-gray-200
                    disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            ← Prev
          </button>
          
          <span className="text-xs sm:text-sm text-gray-400 tabular-nums mx-1">
            {currentPage} / {totalPages}
          </span>
          
          <button
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="min-h-10 min-w-10 px-3 py-2 rounded text-xs sm:text-sm font-medium text-gray-600 
                    bg-gray-50 hover:bg-gray-100 border border-gray-200
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