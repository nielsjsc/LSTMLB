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

  // Value Metrics
  { key: 'war_pit', header: 'WAR', formatter: 'war', align: 'right' },
  { key: 'base', header: 'Value', formatter: 'money', align: 'right' },
  { key: 'contract', header: 'Contract', formatter: 'money', align: 'right' },
  { key: 'surplus', header: 'Surplus', formatter: 'money', align: 'right' }
];

interface CombinedPitchingTableProps {
    data: Array<{
      year: number;
      age: number;
      value: {
        base: number;
        contract: number;
        surplus: number;
      };
      pitching: {
        war_pit: number;
        fip: number;
        siera: number;
        k_pct: number;
        bb_pct: number;
      };
    }>;
  }
  

  const CombinedPitchingTable: React.FC<CombinedPitchingTableProps> = ({ data }) => {
    const formattedData = data.map(row => {
      const { war_pit, ...otherPitchingStats } = row.pitching;
      return {
        year: row.year,
        age: row.age,
        war_pit,
        base: row.value.base,
        contract: row.value.contract,
        surplus: row.value.surplus,
        ...otherPitchingStats
      };
    });
  
    return <StatTable data={formattedData} columns={columns} defaultSort="year" />;
  };

export default CombinedPitchingTable;