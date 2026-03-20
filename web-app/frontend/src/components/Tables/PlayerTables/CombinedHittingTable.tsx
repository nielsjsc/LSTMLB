import React, { useState, useMemo } from 'react';

interface CombinedHittingTableProps {
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
      bat: number;
      bsr: number;
      def_value: number;
      hr: number;
      doubles: number;
      triples: number;
      r: number;
      rbi: number;
      sb: number;
      cs: number;
      xba?: number;
      xslg?: number;
      xwoba?: number;
    };
  }>;
  dividerYear?: number;
  showCareerTotals?: boolean;
  hideXStats?: boolean;
}
interface FormattedRow extends Record<string, any> {
  year: number;
  age: number;
  team: string;
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
  bat: number;
  bsr: number;
  def_value: number;
  hr: number;
  doubles: number;
  triples: number;
  r: number;
  rbi: number;
  sb: number;
  cs: number;
  xba?: number;
  xslg?: number;
  xwoba?: number;
}
const CombinedHittingTable: React.FC<CombinedHittingTableProps> = ({ data, dividerYear, showCareerTotals = false, hideXStats = false }) => {
  const [sortKey, setSortKey] = useState<string>('year');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');


  const headers = [
    { key: 'year', label: 'Year' },
    { key: 'team', label: 'Team' },
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
    ...(!hideXStats ? [
      { key: 'xba', label: 'xBA' },
      { key: 'xslg', label: 'xSLG' },
      { key: 'xwoba', label: 'xwOBA' },
    ] : []),
    { key: 'bat', label: 'Bat' },
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
      case 'bat':
      case 'bsr':
      case 'def_value':
      
        return formatDecimal(value, 1);
      case 'avg':
      case 'obp':
      case 'slg':
      case 'ops':
      case 'woba':
      case 'xba':
      case 'xslg':
      case 'xwoba':
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
    team: row.team ?? '',
    base_value: row.value.base_value,
    contract_value: row.value.contract_value,
    surplus_value: row.value.surplus_value,
    status: row.status,
    ...row.hitting
  })) as FormattedRow[];

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

    const gamesWeightedAvg = (key: string): number | null => {
      let weightedSum = 0;
      let totalWeight = 0;
      for (const row of formattedData) {
        const v = row[key];
        const w = row.g_bat;
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
      g_bat: sum('g_bat'),
      war_bat: sum('war_bat'),
      hr: sum('hr'),
      doubles: sum('doubles'),
      triples: sum('triples'),
      r: sum('r'),
      rbi: sum('rbi'),
      sb: sum('sb'),
      cs: sum('cs'),
      bat: sum('bat'),
      bsr: sum('bsr'),
      def_value: sum('def_value'),
      avg: gamesWeightedAvg('avg'),
      obp: gamesWeightedAvg('obp'),
      slg: gamesWeightedAvg('slg'),
      ops: gamesWeightedAvg('ops'),
      woba: gamesWeightedAvg('woba'),
      wrc_plus: gamesWeightedAvg('wrc_plus'),
      bb_pct_bat: gamesWeightedAvg('bb_pct_bat'),
      k_pct_bat: gamesWeightedAvg('k_pct_bat'),
      xba: gamesWeightedAvg('xba'),
      xslg: gamesWeightedAvg('xslg'),
      xwoba: gamesWeightedAvg('xwoba'),
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
}
export default CombinedHittingTable;