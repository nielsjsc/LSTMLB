import React from 'react';
import StatTable, { Column } from '../BaseTable/StatTable';
const columns: Column[] = [
  // Base/Value
  { key: 'year', header: 'Year', formatter: 'year', align: 'left' },
  { key: 'age', header: 'Age', formatter: 'integer', align: 'right' },
  { key: 'war_pit', header: 'WAR', formatter: 'war', align: 'right' },
  
  // Traditional Stats
  { key: 'era', header: 'ERA', formatter: 'decimal', align: 'right' },
  { key: 'fip', header: 'FIP', formatter: 'decimal', align: 'right' },
  { key: 'siera', header: 'SIERA', formatter: 'decimal', align: 'right' },
  
  // Advanced Stats
  { key: 'k_pct_pit', header: 'K%', formatter: 'percent', align: 'right' },
  { key: 'bb_pct_pit', header: 'BB%', formatter: 'percent', align: 'right' },
  
  // Value
  { key: 'base_value', header: 'Value', formatter: 'money', align: 'right' },
  { key: 'contract_value', header: 'Contract', formatter: 'money', align: 'right' },
  { key: 'surplus_value', header: 'Surplus', formatter: 'money', align: 'right' }
];

interface CombinedPitchingTableProps {
  data: Array<{
      year: number;
      age: number;
      value: {
          base_value: number;
          contract_value: number;
          surplus_value: number;
      };
      pitching: {
          war_pit: number;
          era: number;
          fip: number;
          siera: number;
          k_pct_pit: number;
          bb_pct_pit: number;
          // Ensure interface matches actual data
      };
  }>;
  dividerYear?: number;
}

const CombinedPitchingTable: React.FC<CombinedPitchingTableProps> = ({ data, dividerYear }) => {
  const formattedData = data.map(row => ({
      year: row.year,
      age: row.age,
      base_value: row.value.base_value,
      contract_value: row.value.contract_value,
      surplus_value: row.value.surplus_value,
      ...row.pitching
  }));

  return <StatTable 
    data={formattedData} 
    columns={columns} 
    defaultSort="year"
    defaultSortDirection="asc"
    dividerYear={dividerYear}
  />;
};

export default CombinedPitchingTable;