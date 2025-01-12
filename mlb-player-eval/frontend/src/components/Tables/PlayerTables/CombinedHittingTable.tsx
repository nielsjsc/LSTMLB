import React from 'react';
import StatTable, { Column } from '../BaseTable/StatTable';

const columns: Column[] = [
  // Basic Info
  { key: 'year', header: 'Year', formatter: 'year', align: 'left' },
  { key: 'age', header: 'Age', formatter: 'integer', align: 'right' },

  // Hitting Stats
  { key: 'avg', header: 'AVG', formatter: 'decimal', align: 'right' },
  { key: 'obp', header: 'OBP', formatter: 'decimal', align: 'right' },
  { key: 'slg', header: 'SLG', formatter: 'decimal', align: 'right' },
  { key: 'woba', header: 'wOBA', formatter: 'decimal', align: 'right' },
  { key: 'wrc_plus', header: 'wRC+', formatter: 'integer', align: 'right' },
  { key: 'bb_pct', header: 'BB%', formatter: 'percent', align: 'right' },
  { key: 'k_pct', header: 'K%', formatter: 'percent', align: 'right' },
  { key: 'ev', header: 'EV', formatter: 'decimal', align: 'right' },
  { key: 'off', header: 'Off', formatter: 'decimal', align: 'right' },
  { key: 'def', header: 'Def', formatter: 'decimal', align: 'right' },
  { key: 'bsr', header: 'BsR', formatter: 'decimal', align: 'right' },

  // Value Metrics
  { key: 'war_bat', header: 'WAR', formatter: 'war', align: 'right' },
  { key: 'base', header: 'Value', formatter: 'money', align: 'right' },
  { key: 'contract', header: 'Contract', formatter: 'money', align: 'right' },
  { key: 'surplus', header: 'Surplus', formatter: 'money', align: 'right' }
];

interface CombinedHittingTableProps {
    data: Array<{
      year: number;
      age: number;
      value: {
        base: number;
        contract: number;
        surplus: number;
      };
      hitting: {
        war_bat: number;
        bb_pct: number;
        k_pct: number;
        avg: number;
        obp: number;
        slg: number;
        woba: number;
        wrc_plus: number;
        ev: number;
        off: number;
        bsr: number;
        def: number;
      };
    }>;
  }

  const CombinedHittingTable: React.FC<CombinedHittingTableProps> = ({ data }) => {
    const formattedData = data.map(row => {
      const { war_bat, ...otherHittingStats } = row.hitting;
      return {
        year: row.year,
        age: row.age,
        war_bat,
        base: row.value.base,
        contract: row.value.contract,
        surplus: row.value.surplus,
        ...otherHittingStats
      };
    });
  
    return <StatTable data={formattedData} columns={columns} defaultSort="year" />;
  };

export default CombinedHittingTable;