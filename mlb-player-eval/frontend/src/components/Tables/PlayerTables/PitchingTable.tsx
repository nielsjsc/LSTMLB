import React from 'react';
import StatTable, { Column } from '../BaseTable/StatTable';

const columns: Column[] = [
  { key: 'year', header: 'Year', formatter: 'year', align: 'left' },
  { key: 'age', header: 'Age', formatter: 'integer', align: 'right' },
  // Core Stats
  { key: 'fip', header: 'FIP', formatter: 'decimal', align: 'right' },
  { key: 'siera', header: 'SIERA', formatter: 'decimal', align: 'right' },
  // Control
  { key: 'k_pct', header: 'K%', formatter: 'percent', align: 'right' },
  { key: 'bb_pct', header: 'BB%', formatter: 'percent', align: 'right' },
  // Batted Ball
  { key: 'gb_pct', header: 'GB%', formatter: 'percent', align: 'right' },
  { key: 'fb_pct', header: 'FB%', formatter: 'percent', align: 'right' },
  // Stuff Metrics
  { key: 'stuff_plus', header: 'Stuff+', formatter: 'integer', align: 'right' },
  { key: 'location_plus', header: 'Loc+', formatter: 'integer', align: 'right' },
  { key: 'pitching_plus', header: 'Pit+', formatter: 'integer', align: 'right' },
  // Velocity
  { key: 'fbv', header: 'FBv', formatter: 'decimal', align: 'right' }
];

interface PitchingTableProps {
  data: Array<{
    year: number;
    age: number;
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

const PitchingTable: React.FC<PitchingTableProps> = ({ data }) => {
  const formattedData = data.map(row => ({
    year: row.year,
    age: row.age,
    fip: row.pitching.fip,
    siera: row.pitching.siera,
    k_pct: row.pitching.k_pct,
    bb_pct: row.pitching.bb_pct,
    gb_pct: row.pitching.gb_pct,
    fb_pct: row.pitching.fb_pct,
    stuff_plus: row.pitching.stuff_plus,
    location_plus: row.pitching.location_plus,
    pitching_plus: row.pitching.pitching_plus,
    fbv: row.pitching.fbv
  }));

  return <StatTable data={formattedData} columns={columns} defaultSort="year" />;
};

export default PitchingTable;