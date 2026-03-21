import React, { useState, useMemo } from 'react';

interface CombinedFieldingTableProps {
  data: Array<{
    season: number;
    team: string | null;
    pos: string;
    age: number | null;
    g: number | null;
    gs: number | null;
    inn: number | null;
    sc_total_runs: number | null;
    sc_range_runs: number | null;
    sc_arm_runs: number | null;
    sc_dp_runs: number | null;
    sc_framing_runs: number | null;
    sc_throwing_runs: number | null;
    sc_blocking_runs: number | null;
    drs: number | null;
    uzr: number | null;
    uzr_150: number | null;
    oaa: number | null;
    errors: number | null;
    fp: number | null;
    is_projection: boolean;
  }>;
  showCareerTotals?: boolean;
  hideTraditional?: boolean;
}

interface FormattedRow extends Record<string, any> {
  season: number;
  team: string;
  pos: string;
  age: number | null;
  g: number | null;
  gs: number | null;
  inn: number | null;
  sc_total_runs: number | null;
  sc_range_runs: number | null;
  sc_arm_runs: number | null;
  sc_dp_runs: number | null;
  sc_framing_runs: number | null;
  sc_throwing_runs: number | null;
  sc_blocking_runs: number | null;
  drs: number | null;
  uzr: number | null;
  uzr_150: number | null;
  oaa: number | null;
  errors: number | null;
  fp: number | null;
}

const CombinedFieldingTable: React.FC<CombinedFieldingTableProps> = ({
  data,
  showCareerTotals = false,
  hideTraditional = false,
}) => {
  const [sortKey, setSortKey] = useState<string>('season');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  // Determine if any rows have catcher stats
  const hasCatcherStats = useMemo(
    () => data.some((d) => d.pos === 'C'),
    [data]
  );

  // Determine if any rows have infield stats (DP runs)
  const hasInfieldStats = useMemo(
    () => data.some((d) => ['1B', '2B', '3B', 'SS'].includes(d.pos)),
    [data]
  );

  const headers = useMemo(() => {
    const h = [
      { key: 'season', label: 'Year' },
      { key: 'team', label: 'Team' },
      { key: 'pos', label: 'Pos' },
      { key: 'age', label: 'Age' },
      { key: 'g', label: 'G' },
      { key: 'gs', label: 'GS' },
      { key: 'inn', label: 'Inn' },
      // Statcast run values
      { key: 'sc_total_runs', label: 'Def Runs' },
      { key: 'sc_range_runs', label: 'Range' },
      { key: 'sc_arm_runs', label: 'Arm' },
    ];

    if (hasInfieldStats) {
      h.push({ key: 'sc_dp_runs', label: 'DP' });
    }

    if (hasCatcherStats) {
      h.push(
        { key: 'sc_framing_runs', label: 'Framing' },
        { key: 'sc_throwing_runs', label: 'Throwing' },
        { key: 'sc_blocking_runs', label: 'Blocking' },
      );
    }

    if (!hideTraditional) {
      h.push(
        { key: 'oaa', label: 'OAA' },
        { key: 'drs', label: 'DRS' },
        { key: 'uzr_150', label: 'UZR/150' },
        { key: 'errors', label: 'E' },
        { key: 'fp', label: 'FPct' },
      );
    }

    return h;
  }, [hasCatcherStats, hasInfieldStats, hideTraditional]);

  const formatCell = (key: string, value: any): string => {
    if (value === undefined || value === null) return '—';

    switch (key) {
      case 'fp':
        return Number(value).toFixed(3);
      case 'inn':
        return Number(value).toFixed(1);
      case 'sc_total_runs':
      case 'sc_range_runs':
      case 'sc_arm_runs':
      case 'sc_dp_runs':
      case 'sc_framing_runs':
      case 'sc_throwing_runs':
      case 'sc_blocking_runs':
      case 'uzr':
      case 'uzr_150':
        return Number(value).toFixed(1);
      case 'drs':
      case 'oaa':
      case 'errors':
      case 'g':
      case 'gs':
      case 'age':
        return Math.round(Number(value)).toString();
      default:
        return String(value);
    }
  };

  // Color-code run values (green = positive, red = negative)
  const getRunValueColor = (key: string, value: any): string | undefined => {
    const runKeys = [
      'sc_total_runs', 'sc_range_runs', 'sc_arm_runs', 'sc_dp_runs',
      'sc_framing_runs', 'sc_throwing_runs', 'sc_blocking_runs',
      'drs', 'uzr', 'uzr_150', 'oaa',
    ];
    if (!runKeys.includes(key) || value === null || value === undefined) return undefined;
    const v = Number(value);
    if (v > 0) return '#16a34a'; // green-600
    if (v < 0) return '#dc2626'; // red-600
    return undefined;
  };

  const formattedData: FormattedRow[] = data.map((row) => ({
    ...row,
    team: row.team ?? '',
  }));

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

    return {
      season: 'Total',
      team: '',
      pos: '',
      age: '',
      g: sum('g'),
      gs: sum('gs'),
      inn: sum('inn'),
      sc_total_runs: sum('sc_total_runs'),
      sc_range_runs: sum('sc_range_runs'),
      sc_arm_runs: sum('sc_arm_runs'),
      sc_dp_runs: sum('sc_dp_runs'),
      sc_framing_runs: sum('sc_framing_runs'),
      sc_throwing_runs: sum('sc_throwing_runs'),
      sc_blocking_runs: sum('sc_blocking_runs'),
      drs: sum('drs'),
      oaa: sum('oaa'),
      errors: sum('errors'),
      uzr: sum('uzr'),
      uzr_150: null, // rate stat — don't sum
      fp: null, // rate stat — don't sum
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
              {headers.map((header) => {
                const color = getRunValueColor(header.key, row[header.key]);
                return (
                  <td
                    key={header.key}
                    className="px-1.5 py-2 whitespace-nowrap text-gray-600"
                    style={color ? { color, fontWeight: 500 } : undefined}
                  >
                    {formatCell(header.key, row[header.key])}
                  </td>
                );
              })}
            </tr>
          ))}
          {showCareerTotals && careerTotals && (
            <tr className="bg-gray-50 border-t-2 border-gray-300 text-[11px] font-bold sticky bottom-0">
              {headers.map((header) => {
                const color = getRunValueColor(header.key, careerTotals[header.key]);
                return (
                  <td
                    key={header.key}
                    className="px-1.5 py-2 whitespace-nowrap text-gray-900"
                    style={color ? { color } : undefined}
                  >
                    {formatCell(header.key, careerTotals[header.key])}
                  </td>
                );
              })}
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};

export default CombinedFieldingTable;
