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

  const formatCell = (key: string, value: any, row?: any) => {
    if (value === undefined || value === null) return '-';
    
    if (key === 'name') {
      return (
        <Link to={`/players/${row.real_id}`} className="text-blue-400 hover:text-blue-300">
          {value}
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
  const formattedData = data.players.map(player => ({
    ...player,
    name: { id: player.name.replace(/\s+/g, '-').toLowerCase(), name: player.name }
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
            {data.players.map((row, i) => (
              <tr key={i} className="hover:bg-slate-700/30 text-xs">
                {headers.map((header) => (
                  <td key={header.key} className="px-2 py-1 whitespace-nowrap text-gray-300">
                    {formatCell(header.key, row[header.key], row)}
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

export default TradeValuesTable;