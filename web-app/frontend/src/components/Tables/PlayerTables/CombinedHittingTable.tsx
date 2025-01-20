import React from 'react';
import StatTable, { Column } from '../BaseTable/StatTable';

const columns: Column[] = [
  // Base/Value
  { key: 'year', header: 'Year', formatter: 'year', align: 'left' },
  { key: 'age', header: 'Age', formatter: 'integer', align: 'right' },
  { key: 'war_bat', header: 'WAR', formatter: 'war', align: 'right' },
  
  // Traditional Stats
  { key: 'avg', header: 'AVG', formatter: 'decimal', align: 'right' },
  { key: 'obp', header: 'OBP', formatter: 'decimal', align: 'right' },
  { key: 'slg', header: 'SLG', formatter: 'decimal', align: 'right' },
  { key: 'ops', header: 'OPS', formatter: 'decimal', align: 'right' },
  { key: 'hr', header: 'HR', formatter: 'integer', align: 'right' },
  { key: 'doubles', header: '2B', formatter: 'integer', align: 'right' },
  { key: 'triples', header: '3B', formatter: 'integer', align: 'right' },
  { key: 'r', header: 'R', formatter: 'integer', align: 'right' },
  { key: 'rbi', header: 'RBI', formatter: 'integer', align: 'right' },
  { key: 'sb', header: 'SB', formatter: 'integer', align: 'right' },
  { key: 'cs', header: 'CS', formatter: 'integer', align: 'right' },

  // Advanced Stats  
  { key: 'woba', header: 'wOBA', formatter: 'decimal', align: 'right' },
  { key: 'wrc_plus', header: 'wRC+', formatter: 'integer', align: 'right' },
  { key: 'bb_pct_bat', header: 'BB%', formatter: 'percent', align: 'right' },
  { key: 'k_pct_bat', header: 'K%', formatter: 'percent', align: 'right' },
  { key: 'off', header: 'Off', formatter: 'decimal', align: 'right' },
  { key: 'bsr', header: 'BsR', formatter: 'decimal', align: 'right' },
  { key: 'def_value', header: 'Def', formatter: 'decimal', align: 'right' },
  
  // Value
  { key: 'base_value', header: 'Value', formatter: 'money', align: 'right' },
  { key: 'surplus_value', header: 'Surplus', formatter: 'money', align: 'right' }
];

interface CombinedHittingTableProps {
  data: Array<{
      year: number;
      age: number;
      value: {
          base_value: number;
          contract_value: number;
          surplus_value: number;
      };
      hitting: {
          war_bat: number;
          bb_pct_bat: number;
          k_pct_bat: number;
          avg: number;
          obp: number;
          slg: number;
          ops: number;
          woba: number;
          wrc_plus: number;
          ev: number;
          off: number;
          bsr: number;
          def_value: number;  // Changed from def_val
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

  const CombinedHittingTable: React.FC<CombinedHittingTableProps> = ({ data, dividerYear }) => {
    const formattedData = data.map(row => ({
        year: row.year,
        age: row.age,
        base_value: row.value.base_value,
        contract_value: row.value.contract_value,
        surplus_value: row.value.surplus_value,
        ...row.hitting
    }));

    return <StatTable 
    data={formattedData} 
    columns={columns} 
    defaultSort="year"
    defaultSortDirection="asc"
    dividerYear={dividerYear}  // Pass through
  />;
  };

export default CombinedHittingTable;