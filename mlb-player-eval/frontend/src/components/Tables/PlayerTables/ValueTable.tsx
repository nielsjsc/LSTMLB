import React from 'react';
import StatTable, { Column } from '../BaseTable/StatTable';

const columns: Column[] = [
  { key: 'year', header: 'Year', formatter: 'year', align: 'left' },
  { key: 'war', header: 'WAR', formatter: 'war', align: 'right' },
  { key: 'base', header: 'Value', formatter: 'money', align: 'right' },
  { key: 'contract', header: 'Contract', formatter: 'money', align: 'right' },
  { key: 'surplus', header: 'Surplus', formatter: 'money', align: 'right' }
];

interface ValueTableProps {
  data: Array<{
    year: number;
    war: number;
    value: {
      base: number;
      contract: number;
      surplus: number;
    };
  }>;
}

const ValueTable: React.FC<ValueTableProps> = ({ data }) => {
  const formattedData = data.map(row => ({
    year: row.year,
    war: row.war,
    base: row.value.base,
    contract: row.value.contract,
    surplus: row.value.surplus
  }));

  return <StatTable data={formattedData} columns={columns} defaultSort="year" />;
};

export default ValueTable;