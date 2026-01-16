import React, { useState } from 'react';

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
      <table className="min-w-full divide-y divide-slate-700/50">
        <thead>
          <tr>
            {headers.map((header) => (
              <th
                key={header.key}
                onClick={() => handleSort(header.key)}
                className="px-2 py-1 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-slate-700/50"
              >
                <div className="flex items-center space-x-1">
                  <span>{header.label}</span>
                  {sortKey === header.key && (
                    <span>{sortDirection === 'asc' ? '↑' : '↓'}</span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-700/50">
          {sortedData.map((row, i) => {
            const isProjYear = row.year >= 2025;
            const isFirstProjYear = isProjYear && (!sortedData[i - 1] || sortedData[i - 1].year < 2025);
            
            return (
              <React.Fragment key={i}>
                {isFirstProjYear && (
                  <tr>
                    <td 
                      colSpan={headers.length} 
                      className="bg-emerald-900/20 text-emerald-400 text-xs font-semibold px-2 py-1 text-center border-y border-emerald-600/20"
                    >
                      Projected Stats
                    </td>
                  </tr>
                )}
                <tr 
                  className={`
                    hover:bg-slate-700/30 
                    text-xs text-gray-300
                    ${isProjYear ? 'bg-slate-800/50' : ''}
                  `}
                >
                  {headers.map((header) => (
                    <td key={header.key} className="px-2 py-1 whitespace-nowrap">
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