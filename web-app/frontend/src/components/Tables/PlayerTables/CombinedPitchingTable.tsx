import React, { useState, useMemo } from 'react';

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
      ip: number;
      war_pit: number;
      era: number;
      fip: number;
      k_pct_pit: number;
      bb_pct_pit: number;
      gb_pct?: number;
      fb_pct?: number;
      hr_fb?: number;
      hr_9?: number;
      xera?: number;
    };
  }>;
  dividerYear?: number;
  showCareerTotals?: boolean;
  hideXStats?: boolean;
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
  ip: number;
  war_pit: number;
  era: number;
  fip: number;
  k_pct_pit: number;
  bb_pct_pit: number;
  gb_pct?: number;
  fb_pct?: number;
  hr_fb?: number;
  hr_9?: number;
  xera?: number;
}
const CombinedPitchingTable: React.FC<CombinedPitchingTableProps> = ({ data, dividerYear, showCareerTotals = false, hideXStats = false }) => {
  const [sortKey, setSortKey] = useState<string>('year');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const headers = [
    { key: 'year', label: 'Year' },
    { key: 'team', label: 'Team' },
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
    ...(!hideXStats ? [{ key: 'xera', label: 'xERA' }] : []),
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
      case 'hr_9':
      case 'xera':
        return formatDecimal(value, 2);
      case 'ip':
        return formatDecimal(value, 1);
      case 'k_pct_pit':
      case 'bb_pct_pit':
      case 'gb_pct':
      case 'fb_pct':
      case 'hr_fb':
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

  const careerTotals = useMemo((): Record<string, any> | null => {
    if (formattedData.length === 0) return null;

    const sum = (key: string): number | null => {
      let total = 0;
      let hasValue = false;
      for (const row of formattedData) {
        const v = row[key];
        if (v != null && typeof v === 'number' && !isNaN(v)) {
          total += v;
          hasValue = true;
        }
      }
      return hasValue ? total : null;
    };

    const ipWeightedAvg = (key: string): number | null => {
      let weightedSum = 0;
      let totalWeight = 0;
      for (const row of formattedData) {
        const v = row[key];
        const w = row.ip;
        if (v != null && typeof v === 'number' && !isNaN(v) && w != null && w > 0) {
          weightedSum += v * w;
          totalWeight += w;
        }
      }
      return totalWeight > 0 ? weightedSum / totalWeight : null;
    };

    return {
      year: 'Career',
      age: '',
      team: '',
      status: '',
      g_pit: sum('g_pit'),
      gs: sum('gs'),
      ip: sum('ip'),
      war_pit: sum('war_pit'),
      era: ipWeightedAvg('era'),
      fip: ipWeightedAvg('fip'),
      k_pct_pit: ipWeightedAvg('k_pct_pit'),
      bb_pct_pit: ipWeightedAvg('bb_pct_pit'),
      gb_pct: ipWeightedAvg('gb_pct'),
      fb_pct: ipWeightedAvg('fb_pct'),
      hr_fb: ipWeightedAvg('hr_fb'),
      hr_9: ipWeightedAvg('hr_9'),
      xera: ipWeightedAvg('xera'),
      base_value: sum('base_value'),
      contract_value: sum('contract_value'),
      surplus_value: sum('surplus_value'),
    };
  }, [formattedData]);

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
          <tr className="bg-gray-50 border-b border-gray-200">
            {headers.map((header) => (
              <th
                key={header.key}
                onClick={() => handleSort(header.key)}
                className="px-1.5 py-2 text-left text-[10px] font-semibold text-gray-500 uppercase tracking-wide cursor-pointer hover:text-gray-900 hover:bg-gray-50 select-none transition-colors"
              >
                <div className="flex items-center gap-0.5">
                  <span>{header.label}</span>
                  {sortKey === header.key && (
                    <span className="text-brand-500">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.map((row, i) => (
            <tr
              key={i}
              className={`
                text-[11px] border-b border-gray-100 hover:bg-gray-50 transition-colors
                ${i % 2 === 0 ? 'bg-transparent' : 'bg-gray-50/50'}
              `}
            >
              {headers.map((header) => (
                <td key={header.key} className="px-1.5 py-2 whitespace-nowrap text-gray-600">
                  {formatCell(header.key, row[header.key])}
                </td>
              ))}
            </tr>
          ))}
          {showCareerTotals && careerTotals && (
            <tr className="bg-gray-50 border-t-2 border-gray-300 text-[11px] font-bold sticky bottom-0">
              {headers.map((header) => (
                <td key={header.key} className="px-1.5 py-2 whitespace-nowrap text-gray-900">
                  {formatCell(header.key, careerTotals[header.key])}
                </td>
              ))}
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};

export default CombinedPitchingTable;