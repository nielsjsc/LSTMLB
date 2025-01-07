import React from 'react';
import StatTable, { Column } from '../BaseTable/StatTable';

const columns: Column[] = [
  { key: 'year', header: 'Year', formatter: 'year', align: 'left' },
  { key: 'age', header: 'Age', formatter: 'integer', align: 'right' },
  // Slash Line
  { key: 'avg', header: 'AVG', formatter: 'decimal', align: 'right' },
  { key: 'obp', header: 'OBP', formatter: 'decimal', align: 'right' },
  { key: 'slg', header: 'SLG', formatter: 'decimal', align: 'right' },
  // Advanced
  { key: 'woba', header: 'wOBA', formatter: 'decimal', align: 'right' },
  { key: 'wrc_plus', header: 'wRC+', formatter: 'integer', align: 'right' },
  // Plate Discipline
  { key: 'bb_pct', header: 'BB%', formatter: 'percent', align: 'right' },
  { key: 'k_pct', header: 'K%', formatter: 'percent', align: 'right' },
  // Quality of Contact
  { key: 'ev', header: 'EV', formatter: 'decimal', align: 'right' },
  // Value Components
  { key: 'off', header: 'Off', formatter: 'decimal', align: 'right' },
  { key: 'def', header: 'Def', formatter: 'decimal', align: 'right' },
  { key: 'bsr', header: 'BsR', formatter: 'decimal', align: 'right' }
];

interface HittingTableProps {
  data: Array<{
    year: number;
    hitting: {
      age: number;
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

const HittingTable: React.FC<HittingTableProps> = ({ data }) => {
  const formattedData = data.map(row => ({
    year: row.year,
    age: row.hitting.age,
    bb_pct: row.hitting.bb_pct,
    k_pct: row.hitting.k_pct,
    avg: row.hitting.avg,
    obp: row.hitting.obp,
    slg: row.hitting.slg,
    woba: row.hitting.woba,
    wrc_plus: row.hitting.wrc_plus,
    ev: row.hitting.ev,
    off: row.hitting.off,
    bsr: row.hitting.bsr,
    def: row.hitting.def
  }));

  return <StatTable data={formattedData} columns={columns} defaultSort="year" />;
};

export default HittingTable;