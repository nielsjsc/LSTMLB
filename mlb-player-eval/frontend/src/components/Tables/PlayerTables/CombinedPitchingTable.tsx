import React from 'react';
import StatTable, { Column } from '../BaseTable/StatTable';

const columns: Column[] = [
  // Basic Info
  { key: 'year', header: 'Year', formatter: 'year', align: 'left' },
  { key: 'age', header: 'Age', formatter: 'integer', align: 'right' },

  // Core Stats
  { key: 'fip', header: 'FIP', formatter: 'decimal', align: 'right' },
  { key: 'siera', header: 'SIERA', formatter: 'decimal', align: 'right' },
  { key: 'k_pct', header: 'K%', formatter: 'percent', align: 'right' },
  { key: 'bb_pct', header: 'BB%', formatter: 'percent', align: 'right' },
  
  // Batted Ball
  { key: 'gb_pct', header: 'GB%', formatter: 'percent', align: 'right' },
  { key: 'fb_pct', header: 'FB%', formatter: 'percent', align: 'right' },
  
  // Plus Stats
  { key: 'stuff_plus', header: 'Stuff+', formatter: 'integer', align: 'right' },
  { key: 'location_plus', header: 'Loc+', formatter: 'integer', align: 'right' },
  { key: 'pitching_plus', header: 'Pit+', formatter: 'integer', align: 'right' },
  { key: 'fbv', header: 'FBv', formatter: 'decimal', align: 'right' },

  // Value Metrics
  { key: 'war', header: 'WAR', formatter: 'war', align: 'right' },
  { key: 'base', header: 'Value', formatter: 'money', align: 'right' },
  { key: 'contract', header: 'Contract', formatter: 'money', align: 'right' },
  { key: 'surplus', header: 'Surplus', formatter: 'money', align: 'right' }
];

interface CombinedPitchingTableProps {
  data: Array<{
    year: number;
    age: number;
    war: number;
    value: {
      base: number;
      contract: number;
      surplus: number;
    };
    pitching: {
      fip: number;
      siera: number;
      k_pct: number;
      bb_pct: number;
      gb_pct: number;
      fb_pct: number;
      stuff_plus: number;
      location_plus: number;
      pitching_plus: number;
      fbv: number;
    };
  }>;
}

const CombinedPitchingTable: React.FC<CombinedPitchingTableProps> = ({ data }) => {
  const formattedData = data.map(row => ({
    year: row.year,
    age: row.age,
    war: row.war,
    base: row.value.base,
    contract: row.value.contract,
    surplus: row.value.surplus,
    ...row.pitching
  }));

  return <StatTable data={formattedData} columns={columns} defaultSort="year" />;
};

export default CombinedPitchingTable;