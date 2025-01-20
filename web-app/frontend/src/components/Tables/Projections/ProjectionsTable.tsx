import React from 'react';
import { Link } from 'react-router-dom';
import StatTable, { Column } from '../BaseTable/StatTable';
import { ProjectionResponse } from '../../../services/api';
import { StatFormatterType } from '../../../utils/statFormatters';

const hitterColumns: Column[] = [
  { 
    key: 'name', 
    header: 'Name', 
    align: 'left',
    renderCell: (value: { id: string, name: string }) => (
      <Link to={`/players/${value.id}`} className="text-blue-600 hover:underline">
        {value.name}
      </Link>
    )
  },
  { key: 'team', header: 'Team', formatter: 'string' as StatFormatterType, align: 'left' },
  { key: 'position', header: 'Pos', formatter: 'string' as StatFormatterType, align: 'left' },
  { key: 'age', header: 'Age', formatter: 'number' as StatFormatterType, align: 'right' },
  // Hitting Stats
  { key: 'war_bat', header: 'WAR', formatter: 'war' as StatFormatterType, align: 'right' },
  { key: 'avg', header: 'AVG', formatter: 'average' as StatFormatterType, align: 'right' },
  { key: 'obp', header: 'OBP', formatter: 'average' as StatFormatterType, align: 'right' },
  { key: 'slg', header: 'SLG', formatter: 'average' as StatFormatterType, align: 'right' },
  { key: 'ops', header: 'OPS', formatter: 'average' as StatFormatterType, align: 'right' },
  { key: 'woba', header: 'wOBA', formatter: 'average' as StatFormatterType, align: 'right' },
  { key: 'wrc_plus', header: 'wRC+', formatter: 'number' as StatFormatterType, align: 'right' },
  { key: 'hr', header: 'HR', formatter: 'number' as StatFormatterType, align: 'right' },
  { key: 'sb', header: 'SB', formatter: 'number' as StatFormatterType, align: 'right' },
  // Value
  { key: 'base_value', header: 'Value', formatter: 'money' as StatFormatterType, align: 'right' },
  { key: 'surplus_value', header: 'Surplus', formatter: 'money' as StatFormatterType, align: 'right' }
];

// Update pitcher columns similarly
const pitcherColumns: Column[] = [
  { 
    key: 'name', 
    header: 'Name', 
    align: 'left',
    renderCell: (value: { id: string, name: string }) => (
      <Link to={`/players/${value.id}`} className="text-blue-600 hover:underline">
        {value.name}
      </Link>
    )
  },
  { key: 'team', header: 'Team', formatter: 'string' as StatFormatterType, align: 'left' },
  { key: 'position', header: 'Pos', formatter: 'string' as StatFormatterType, align: 'left' },
  { key: 'age', header: 'Age', formatter: 'number' as StatFormatterType, align: 'right' },
  // Pitching Stats
  { key: 'war_pit', header: 'WAR', formatter: 'war' as StatFormatterType, align: 'right' },
  { key: 'era', header: 'ERA', formatter: 'decimal' as StatFormatterType, align: 'right' },
  { key: 'fip', header: 'FIP', formatter: 'decimal' as StatFormatterType, align: 'right' },
  { key: 'siera', header: 'SIERA', formatter: 'decimal' as StatFormatterType, align: 'right' },
  { key: 'k_pct_pit', header: 'K%', formatter: 'percent' as StatFormatterType, align: 'right' },
  { key: 'bb_pct_pit', header: 'BB%', formatter: 'percent' as StatFormatterType, align: 'right' },
  // Value
  { key: 'base_value', header: 'Value', formatter: 'money' as StatFormatterType, align: 'right' },
  { key: 'surplus_value', header: 'Surplus', formatter: 'money' as StatFormatterType, align: 'right' }
];
interface ProjectionsTableProps {
  data: ProjectionResponse;
  playerType: 'hitter' | 'pitcher';
}

const ProjectionsTable: React.FC<ProjectionsTableProps> = ({ data, playerType }) => {
  const columns = playerType === 'hitter' ? hitterColumns : pitcherColumns;

  const formattedData = data.players.map(player => ({
    id: player.id,
    name: { id: player.id, name: player.name },  // Pass object with id and name
    team: player.team,
    position: player.position,
    age: player.age,
    ...(playerType === 'hitter' ? player.hitting : player.pitching),
    base_value: player.value.base_value,
    surplus_value: player.value.surplus_value
  }));

  return (
    <div className="mt-4">
      <StatTable 
        data={formattedData} 
        columns={columns} 
        defaultSort={playerType === 'hitter' ? 'war_bat' : 'war_pit'} 
      />
    </div>
  );
};

export default ProjectionsTable;