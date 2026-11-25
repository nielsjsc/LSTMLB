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
  fg_id: number;
  mlbam_id: number | null;
  name: { id: string; name: string };
  team: string;
  position: string;
  control_war: number;
  control_production_value: number;
  control_salary_cost: number;
  trade_value: number;
  trade_value_p10?: number | null;
  trade_value_p90?: number | null;
  control_through: number;
}

const TradeValuesTable: React.FC<TradeValuesTableProps> = ({
  data,
  currentPage,
  totalPages,
  onPageChange,
  onSort,
  sortBy,
  sortDirection,
}) => {
  const desktopHeaders = [
    { key: 'name', label: 'Name', sortable: false },
    { key: 'team', label: 'Team', sortable: false },
    { key: 'position', label: 'Pos', sortable: false },
    { key: 'control_war', label: 'Control WAR', sortable: true },
    { key: 'control_production_value', label: 'Production ($M)', sortable: true },
    { key: 'control_salary_cost', label: 'Contract ($M)', sortable: true },
    { key: 'trade_value', label: 'Trade Value ($M)', sortable: true },
    { key: 'trade_value_p90', label: 'P10 – P90 ($M)', sortable: true },
    { key: 'control_through', label: 'Control Thru', sortable: true },
  ];

  const formatValue = (value: number | undefined | null) => {
    if (value === undefined || value === null || isNaN(value)) return '-';
    return (value / 1_000_000).toFixed(1);
  };

  const formatDecimal = (value: number | undefined | null, digits: number = 1) => {
    if (value === undefined || value === null || isNaN(value)) return '-';
    return value.toFixed(digits);
  };

  const formatRangeShort = (p10: number | undefined | null, p90: number | undefined | null) => {
    if (p10 == null || p90 == null || isNaN(p10) || isNaN(p90)) return '-';
    return `${formatValue(p10)} to ${formatValue(p90)}`;
  };

  const formatCell = (key: string, value: any, row: FormattedTradeValueRow) => {
    if (key === 'name') {
      return (
        <Link to={`/players/${row.mlbam_id || row.fg_id}`} className="text-accent-blue hover:text-blue-300 font-medium">
          {value.name}
        </Link>
      );
    }

    if (key === 'trade_value_p90') {
      return (
        <span className="text-gray-400 tabular-nums">
          {formatRangeShort(row.trade_value_p10, row.trade_value_p90)}
        </span>
      );
    }

    if (value === undefined || value === null) return '-';

    switch (key) {
      case 'avg_war':
      case 'control_war':
        return formatDecimal(value);
      case 'control_salary_cost':
      case 'avg_salary_cost':
      case 'trade_value':
      case 'control_production_value':
        return formatValue(value);
      case 'control_through':
        return value;
      default:
        return value.toString();
    }
  };

  const formattedData: FormattedTradeValueRow[] = data.players.map((player) => ({
    ...player,
    name: {
      id: String(player.mlbam_id || player.fg_id),
      name: player.name,
    },
  }));

  return (
    <div>
      {/* Mobile Card View */}
      <div className="md:hidden space-y-3">
        {formattedData.map((row, i) => (
          <Link
            key={row.fg_id || i}
            to={`/players/${row.mlbam_id || row.fg_id}`}
            className="block p-4 rounded-lg border border-gray-200 bg-white hover:bg-gray-50 transition-colors"
          >
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="font-semibold text-gray-900 text-sm">{row.name.name}</h3>
                <p className="text-xs text-gray-500 mt-0.5">{row.team} • {row.position}</p>
              </div>
              <div className="text-right">
                <span className="text-xs font-semibold text-brand-500 bg-brand-500/10 px-2.5 py-1 rounded whitespace-nowrap">
                  ${formatValue(row.trade_value)}M
                </span>
                {row.trade_value_p10 != null && row.trade_value_p90 != null && (
                  <p className="text-[10px] text-gray-400 mt-1 tabular-nums">
                    ${formatValue(row.trade_value_p10)}M – ${formatValue(row.trade_value_p90)}M
                  </p>
                )}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <p className="text-gray-500">Control WAR</p>
                <p className="font-semibold text-gray-900">{formatDecimal(row.control_war)}</p>
              </div>
              <div>
                <p className="text-gray-500">Production</p>
                <p className="font-semibold text-gray-900">${formatValue(row.control_production_value)}M</p>
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
                  onClick={() => header.sortable && onSort(header.key)}
                  className={`px-2 py-1.5 text-left text-[10px] font-semibold text-gray-500 uppercase tracking-wide select-none transition-colors ${
                    header.sortable ? 'cursor-pointer hover:text-gray-900 hover:bg-gray-100' : ''
                  }`}
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
              <tr
                key={row.fg_id || i}
                className={`border-b border-gray-100 hover:bg-gray-50 transition-colors ${
                  i % 2 === 0 ? 'bg-transparent' : 'bg-gray-50/50'
                }`}
              >
                {desktopHeaders.map((header) => (
                  <td key={header.key} className="px-2 py-1.5 whitespace-nowrap text-xs text-gray-600">
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
            className="min-h-10 min-w-10 px-3 py-2 rounded text-xs sm:text-sm font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 border border-gray-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            ← Prev
          </button>

          <span className="text-xs sm:text-sm text-gray-400 tabular-nums mx-1">
            {currentPage} / {totalPages}
          </span>

          <button
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="min-h-10 min-w-10 px-3 py-2 rounded text-xs sm:text-sm font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 border border-gray-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
};

export default TradeValuesTable;
