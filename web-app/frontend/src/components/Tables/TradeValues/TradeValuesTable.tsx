import React from 'react';
import { Link } from 'react-router-dom';
import { TradeValueRankingsResponse } from '../../../services/api';

interface TradeValuesTableProps {
  data: TradeValueRankingsResponse;
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  onSort: (key: string) => void;
  sortBy: string;
  sortDirection: 'asc' | 'desc';
}
interface FormattedTradeValueRow extends Record<string, any> {
  real_id: number;
  mlb_id: number | null;
  name: { id: string; name: string };
  team: string;
  position: string;
  contract_war: number;
  contract_base_value: number;
  total_contract: number;
  trade_value: number;
  control_through: number;
}
const TradeValuesTable: React.FC<TradeValuesTableProps> = ({
  data,
  currentPage,
  totalPages,
  onPageChange,
  onSort,
  sortBy,
  sortDirection
}) => {
  const headers = [
    { key: 'name', label: 'Name' },
    { key: 'team', label: 'Team' },
    { key: 'position', label: 'Pos' },
    { key: 'contract_war', label: 'WAR  Under Contract ' },
    { key: 'contract_base_value', label: 'Production Value Under Contract' },
    { key: 'total_contract', label: 'Contract $' },
    { key: 'trade_value', label: 'Trade Value' },
    { key: 'control_through', label: 'Control Through' }
  ];

  const formatValue = (value: number | undefined | null) => {
    if (value === undefined || value === null) return '-';
    return (value / 1000000).toFixed(1);
  };

  const formatDecimal = (value: number | undefined | null, digits: number = 1) => {
    if (value === undefined || value === null) return '-';
    return value.toFixed(digits);
  };

  const formatCell = (key: string, value: any, row: FormattedTradeValueRow) => {
    if (value === undefined || value === null) return '-';
    
    if (key === 'name') {
      return (
        <Link to={`/players/${row.mlb_id || row.real_id}`} className="text-accent-blue hover:text-blue-300 font-medium">
          {value.name}
        </Link>
      );
    }

    switch (key) {
      case 'total_war':
      case 'avg_war':
        return formatDecimal(value);
      case 'total_contract':
      case 'avg_contract':
      case 'trade_value':
      case 'contract_base_value':
        return formatValue(value);
      case 'control_through':
        return value;
      default:
        return value.toString();
    }
  };
  const formattedData: FormattedTradeValueRow[] = data.players.map(player => ({
    ...player,
    name: { 
      id: String(player.mlb_id || player.real_id), 
      name: player.name 
    }
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
                    {formatCell(header.key, row[header.key], row)}
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

export default TradeValuesTable;