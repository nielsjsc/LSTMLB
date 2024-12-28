export interface Player {
  id: string;
  name: string;
  team: string;
  position: string;
  value: number;
  war: number;
  salary: number;
  projectedWAR: number;
}

export const teams = [
  'LAA', 'HOU', 'OAK', 'TOR', 'ATL', 
  'MIL', 'STL', 'CHC', 'ARI', 'LAD', 
  'SF', 'TEX', 'CHW', 'CLE', 'DET',
  'KC', 'MIN', 'NYY', 'BAL', 'BOS', 
  'TB', 'MIA', 'NYM', 'PHI', 'WSH',
  'CIN', 'COL', 'SD', 'PIT', 'SEA'
];

export const players: Player[] = [
  {
    id: '1',
    name: 'Mike Trout',
    team: 'LAA',
    position: 'CF',
    value: 45.2,
    war: 6.8,
    salary: 37.1,
    projectedWAR: 6.2
  },
  {
    id: '2',
    name: 'Juan Soto',
    team: 'NYY',
    position: 'RF',
    value: 42.8,
    war: 5.9,
    salary: 31.0,
    projectedWAR: 5.7
  },
  {
    id: '3',
    name: 'Ronald Acuña Jr.',
    team: 'ATL',
    position: 'RF',
    value: 50.1,
    war: 7.2,
    salary: 17.0,
    projectedWAR: 6.8
  },
  {
    id: '4',
    name: 'Mookie Betts',
    team: 'LAD',
    position: 'RF',
    value: 44.5,
    war: 6.6,
    salary: 25.0,
    projectedWAR: 6.1
  },
  {
    id: '5',
    name: 'Aaron Judge',
    team: 'NYY',
    position: 'RF',
    value: 41.2,
    war: 6.1,
    salary: 40.0,
    projectedWAR: 5.8
  }
]