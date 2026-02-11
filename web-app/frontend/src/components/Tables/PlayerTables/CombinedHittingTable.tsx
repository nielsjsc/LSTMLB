import React, { useState } from 'react';
import { CURRENT_YEAR } from '../../../config';

interface CombinedHittingTableProps {
  data: Array<{
    year: number;
    age: number;
    status: string;
    value: {
      base_value: number;
      contract_value: number;
      surplus_value: number;
    };
    hitting: {
      g_bat: number;
      war_bat: number;
      bb_pct_bat: number;
      k_pct_bat: number;
      avg: number;
      obp: number;
      slg: number;
      ops: number;
      woba: number;
      wrc_plus: number;
      off: number;
      bsr: number;
      def_value: number;
      hr: number;
      doubles: number;
      triples: number;
      r: number;
      rbi: number;
      sb: number;
      cs: number;
    };
  }>;
  dividerYear?: number;
}
interface FormattedRow extends Record<string, any> {
  year: number;
  age: number;
  status: string;
  base_value: number;
  contract_value: number;
  surplus_value: number;
  g_bat: number;
  war_bat: number;
  bb_pct_bat: number;
  k_pct_bat: number;
  avg: number;
  obp: number;
  slg: number;
  ops: number;
  woba: number;
  wrc_plus: number;
  off: number;
  bsr: number;
  def_value: number;
  hr: number;
  doubles: number;
  triples: number;
  r: number;
  rbi: number;
  sb: number;
  cs: number;
}
const CombinedHittingTable: React.FC<CombinedHittingTableProps> = ({ data, dividerYear }) => {
  const [sortKey, setSortKey] = useState<string>('year');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');


  const headers = [
    { key: 'year', label: 'Year' },
    { key: 'age', label: 'Age' },
    { key: 'g_bat', label: 'G' },
    { key: 'war_bat', label: 'WAR' },
    // Traditional Stats
    { key: 'avg', label: 'AVG' },
    { key: 'obp', label: 'OBP' },
    { key: 'slg', label: 'SLG' },
    { key: 'ops', label: 'OPS' },
    { key: 'hr', label: 'HR' },
    { key: 'doubles', label: '2B' },
    { key: 'triples', label: '3B' },
    { key: 'r', label: 'R' },
    { key: 'rbi', label: 'RBI' },
    { key: 'sb', label: 'SB' },
    { key: 'cs', label: 'CS' },
    // Advanced Stats
    { key: 'woba', label: 'wOBA' },
    { key: 'wrc_plus', label: 'wRC+' },
    { key: 'bb_pct_bat', label: 'BB%' },
    { key: 'k_pct_bat', label: 'K%' },
    { key: 'off', label: 'Off' },
    { key: 'bsr', label: 'BsR' },
    { key: 'def_value', label: 'Def' },
    // Value
    { key: 'base_value', label: 'Value ($M)' },
    { key: 'contract_value', label: 'Contract ($M)' },
    { key: 'surplus_value', label: 'Surplus ($M)' },
    { key: 'status', label: 'Status' }  ];

  const formatValue = (value: number | undefined | null) => {
    if (value === undefined || value === null) return '-';
    return (value / 1000000).toFixed(1);
  };

  const formatPercent = (value: number | undefined | null) => {
    if (value === undefined || value === null) return '-';
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatDecimal = (value: number | undefined | null, digits: number = 3) => {
    if (value === undefined || value === null) return '-';
    return value.toFixed(digits);
  };

  const formatCell = (key: string, value: any) => {
    if (value === undefined || value === null) return '-';
    
    switch (key) {
      case 'wrc_plus':
        return formatDecimal(value, 0);
      case 'war_bat':
      case 'off':
      case 'bsr':
      case 'def_value':
      
        return formatDecimal(value, 1);
      case 'avg':
      case 'obp':
      case 'slg':
      case 'ops':
      case 'woba':
        return formatDecimal(value, 3);
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

  const formattedData = data.map(row => ({
    ...row,
    base_value: row.value.base_value,
    contract_value: row.value.contract_value,
    surplus_value: row.value.surplus_value,
    status: row.status,
    ...row.hitting
  })) as FormattedRow[];

  const handleSort = (key: string) => {
    const newDirection = sortKey === key && sortDirection === 'asc' ? 'desc' : 'asc';
    setSortKey(key);
    setSortDirection(newDirection);
  };

  const sortedData = [...formattedData].sort((a: FormattedRow, b: FormattedRow) => {
    const aValue = a[sortKey] ?? -Infinity;
    const bValue = b[sortKey] ?? -Infinity;
    return sortDirection === 'asc' 
      ? (aValue > bValue ? 1 : -1)
      : (bValue > aValue ? 1 : -1);
  });

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full">
        <thead className="sticky top-0 z-10">
          <tr className="bg-surface-850 border-b border-white/[0.06]">
            {headers.map((header) => (
              <th
                key={header.key}
                onClick={() => handleSort(header.key)}
                className="px-1.5 py-2 text-left text-[10px] font-semibold text-surface-400 uppercase tracking-wide cursor-pointer hover:text-white hover:bg-white/[0.04] select-none transition-colors"
              >
                <div className="flex items-center gap-0.5">
                  <span>{header.label}</span>
                  {sortKey === header.key && (
                    <span className="text-brand-400">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.map((row, i) => {
            const isProjYear = row.year >= CURRENT_YEAR;
            const isFirstProjYear = isProjYear && (!sortedData[i - 1] || sortedData[i - 1].year < CURRENT_YEAR);
            
            return (
              <React.Fragment key={i}>
                {isFirstProjYear && (
                  <tr>
                    <td 
                      colSpan={headers.length} 
                      className="bg-brand-400/10 text-brand-400 text-xs font-semibold px-3 py-1.5 text-center border-y border-brand-400/20"
                    >
                      ▾ Projected Stats
                    </td>
                  </tr>
                )}
                <tr 
                  className={`
                    text-[11px] border-b border-white/[0.03] hover:bg-white/[0.04] transition-colors
                    ${isProjYear ? 'bg-brand-400/[0.02]' : (i % 2 === 0 ? 'bg-transparent' : 'bg-white/[0.015]')}
                  `}
                >
                  {headers.map((header) => (
                    <td key={header.key} className="px-1.5 py-2 whitespace-nowrap text-surface-300 font-mono">
                      {formatCell(header.key, row[header.key])}
                    </td>
                  ))}
                </tr>
              </React.Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
export default CombinedHittingTable;