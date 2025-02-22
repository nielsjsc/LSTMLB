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
        <table className="min-w-full divide-y divide-slate-700/50">
          <thead>
            <tr>
              {renderHeader().map((header) => (
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
            {data.players.map((player, i) => (
              <tr key={i} className="hover:bg-slate-700/30 text-xs">
                <td className="px-2 py-1 whitespace-nowrap">
                  {player.has_mlb && player.IDfg ? (
                    <Link 
                      to={`/players/${player.IDfg}`} 
                      className="text-blue-400 hover:text-blue-300"
                    >
                      {player.name}
                    </Link>
                  ) : (
                    <span className="text-gray-300">
                      {player.name}
                    </span>
                  )}
                </td>
                <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.org}</td>
                <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.position}</td>
                <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.age}</td>
                <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.fv}</td>
                <td className="px-2 py-1 whitespace-nowrap text-gray-300">{formatValue(player.value)}</td>
                <td className="px-2 py-1 whitespace-nowrap text-gray-300">{formatRanking(player.composite)}</td>

                {playerType === 'hitter' ? (
                  <>
                    <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.hit || '-'}</td>
                    <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.game || '-'}</td>
                    <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.raw || '-'}</td>
                    <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.speed || '-'}</td>
                  </>
                ) : (
                  <>
                    <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.fastball || '-'}</td>
                    <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.slider || '-'}</td>
                    <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.curve || '-'}</td>
                    <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.change || '-'}</td>
                    <td className="px-2 py-1 whitespace-nowrap text-gray-300">{player.command || '-'}</td>
                  </>
                )}
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
              {Math.min(currentPage * 50, data.count)}
            </span> of{' '}
            <span className="font-medium">{data.count}</span> results
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
};

export default ProspectsTable;