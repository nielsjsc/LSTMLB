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
  const mobileHeaders = [
    { key: 'name', label: 'Name' },
    { key: 'team', label: 'Team' },
    { key: 'position', label: 'Pos' },
    { key: 'trade_value', label: 'Trade Value' }
  ];

  const desktopHeaders = [
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
      case 'contract_war':
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
      {/* Mobile Card View */}
      <div className="md:hidden space-y-3">
        {formattedData.map((row, i) => (
          <Link
            key={i}
            to={`/players/${row.mlb_id || row.real_id}`}
            className="block p-4 rounded-lg border border-gray-200 bg-white hover:bg-gray-50 transition-colors"
          >
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="font-semibold text-gray-900 text-sm">{row.name.name}</h3>
                <p className="text-xs text-gray-500 mt-0.5">{row.team} • {row.position}</p>
              </div>
              <span className="text-xs font-semibold text-brand-500 bg-brand-500/10 px-2.5 py-1 rounded whitespace-nowrap">
                ${formatValue(row.trade_value)}M
              </span>
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <p className="text-gray-500">WAR</p>
                <p className="font-semibold text-gray-900">{formatDecimal(row.contract_war)}</p>
              </div>
              <div>
                <p className="text-gray-500">Production</p>
                <p className="font-semibold text-gray-900">${formatValue(row.contract_base_value)}M</p>
              </div>
            </div>
          </Link>
        ))}
      </div>

      {/* Desktop Table View */}
      <div className="hidden md:block overflow-x-auto">
        <table className="min-w-full">
          <thead className="sticky top-0 z-10">
            <tr className="bg-gray-50 border-b border-gray-200">
              {desktopHeaders.map((header) => (
                <th
                  key={header.key}
                  onClick={() => onSort(header.key)}
                  className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide cursor-pointer hover:text-gray-900 hover:bg-gray-100 select-none transition-colors"
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
                {desktopHeaders.map((header) => (
                  <td key={header.key} className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                    {formatCell(header.key, row[header.key], row)}
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

export default TradeValuesTable;