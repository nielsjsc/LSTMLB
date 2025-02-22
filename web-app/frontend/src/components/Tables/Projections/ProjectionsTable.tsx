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
    { key: 'siera', label: 'SIERA' },
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
        <Link to={`/players/${value.id}`} className="text-blue-400 hover:text-blue-300">
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
      case 'siera':
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

  const formattedData = data.players.map(player => ({
    id: player.real_id,
    name: { id: player.real_id, name: player.name },
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
        <table className="min-w-full divide-y divide-slate-700/50">
          <thead>
            <tr>
              {headers.map((header) => (
                <th
                  key={header.key}
                  onClick={() => onSort(header.key)}
                  className="px-2 py-1 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-slate-700/50"
                >
                  <div className="flex items-center space-x-1">
                    <span>{header.label}</span>
                    {sortBy === header.key && (
                      <span>{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/50">
            {formattedData.map((row, i) => (
              <tr key={i} className="hover:bg-slate-700/30 text-xs">
                {headers.map((header) => (
                  <td key={header.key} className="px-2 py-1 whitespace-nowrap text-gray-300">
                    {formatCell(header.key, row[header.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination Controls */}
      <div className="mt-4 flex items-center justify-between px-4">
        <div>
          <p className="text-sm text-gray-400">
            Showing <span className="font-medium">{(currentPage - 1) * 50 + 1}</span> to{' '}
            <span className="font-medium">
              {Math.min(currentPage * 50, data.total_count)}
            </span> of{' '}
            <span className="font-medium">{data.total_count}</span> results
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage === 1}
            className="px-3 py-1 border border-slate-600 rounded-md text-sm font-medium 
                    text-gray-300 hover:bg-slate-700/50 
                    disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          
          <span className="text-sm text-gray-400">
            Page {currentPage} of {totalPages}
          </span>
          
          <button
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="px-3 py-1 border border-slate-600 rounded-md text-sm font-medium 
                    text-gray-300 hover:bg-slate-700/50 
                    disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
export default ProjectionsTable;