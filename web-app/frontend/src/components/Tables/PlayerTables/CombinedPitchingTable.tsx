import React, { useState } from 'react';
import { CURRENT_YEAR } from '../../../config';

interface CombinedPitchingTableProps {
  data: Array<{
    year: number;
    age: number;
    team?: string;
    status: string;
    value: {
      base_value: number;
      contract_value: number;
      surplus_value: number;
    };
    pitching: {
      g_pit: number;
      gs: number;
      war_pit: number;
      era: number;
      fip: number;
      siera: number;
      k_pct_pit: number;
      bb_pct_pit: number;
    };
  }>;
  dividerYear?: number;
}
interface FormattedPitchingRow extends Record<string, any> {
  year: number;
  age: number;
  team: string;
  status: string;
  base_value: number;
  contract_value: number;
  surplus_value: number;
  g_pit: number;
  gs: number;
  war_pit: number;
  era: number;
  fip: number;
  siera: number;
  k_pct_pit: number;
  bb_pct_pit: number;
}
const CombinedPitchingTable: React.FC<CombinedPitchingTableProps> = ({ data, dividerYear }) => {
  const [sortKey, setSortKey] = useState<string>('year');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const headers = [
    { key: 'year', label: 'Year' },
    { key: 'team', label: 'Team' },
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

  const formatValue = (value: number | undefined | null) => {
    if (value === undefined || value === null) return '-';
    return (value / 1000000).toFixed(1);
  };

  const formatPercent = (value: number | undefined | null) => {
    if (value === undefined || value === null) return '-';
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatDecimal = (value: number | undefined | null, digits: number = 2) => {
    if (value === undefined || value === null) return '-';
    return value.toFixed(digits);
  };

  const formatCell = (key: string, value: any) => {
    if (value === undefined || value === null) return '-';
    
    switch (key) {
      case 'war_pit':
        return formatDecimal(value, 1);
      case 'era':
      case 'fip':
      case 'siera':
        return formatDecimal(value, 2);
      case 'k_pct_pit':
      case 'bb_pct_pit':
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
    team: row.team ?? '',
    base_value: row.value.base_value,
    contract_value: row.value.contract_value,
    surplus_value: row.value.surplus_value,
    status: row.status,
    ...row.pitching
  })) as FormattedPitchingRow[];

  const handleSort = (key: string) => {
    const newDirection = sortKey === key && sortDirection === 'asc' ? 'desc' : 'asc';
    setSortKey(key);
    setSortDirection(newDirection);
  };

  const sortedData = [...formattedData].sort((a: FormattedPitchingRow, b: FormattedPitchingRow) => {
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
};

export default CombinedPitchingTable;