export interface TeamInfo {
  division: string;
  name: string;
}
interface TeamDivision {
    [key: string]: {
      division: string;
      name: string;
    };
  }
export const teamDivisions: TeamDivision = {
  'bal': { division: 'AL East', name: 'Baltimore Orioles' },
  'bos': { division: 'AL East', name: 'Boston Red Sox' },
  'nyy': { division: 'AL East', name: 'New York Yankees' },
  'tb': { division: 'AL East', name: 'Tampa Bay Rays' },
  'tor': { division: 'AL East', name: 'Toronto Blue Jays' },
  'cws': { division: 'AL Central', name: 'Chicago White Sox' },
  'cle': { division: 'AL Central', name: 'Cleveland Guardians' },
  'det': { division: 'AL Central', name: 'Detroit Tigers' },
  'kc': { division: 'AL Central', name: 'Kansas City Royals' },
  'min': { division: 'AL Central', name: 'Minnesota Twins' },
  'hou': { division: 'AL West', name: 'Houston Astros' },
  'laa': { division: 'AL West', name: 'Los Angeles Angels' },
  'ath': { division: 'AL West', name: 'Sacramento Athletics' },
  'sea': { division: 'AL West', name: 'Seattle Mariners' },
  'tex': { division: 'AL West', name: 'Texas Rangers' },
  'atl': { division: 'NL East', name: 'Atlanta Braves' },
  'mia': { division: 'NL East', name: 'Miami Marlins' },
  'nym': { division: 'NL East', name: 'New York Mets' },
  'phi': { division: 'NL East', name: 'Philadelphia Phillies' },
  'wsh': { division: 'NL East', name: 'Washington Nationals' },
  'chc': { division: 'NL Central', name: 'Chicago Cubs' },
  'cin': { division: 'NL Central', name: 'Cincinnati Reds' },
  'mil': { division: 'NL Central', name: 'Milwaukee Brewers' },
  'pit': { division: 'NL Central', name: 'Pittsburgh Pirates' },
  'stl': { division: 'NL Central', name: 'St. Louis Cardinals' },
  'ari': { division: 'NL West', name: 'Arizona Diamondbacks' },
  'col': { division: 'NL West', name: 'Colorado Rockies' },
  'lad': { division: 'NL West', name: 'Los Angeles Dodgers' },
  'sd': { division: 'NL West', name: 'San Diego Padres' },
  'sf': { division: 'NL West', name: 'San Francisco Giants' }
};

export const sortTeamsByDivision = (teams: string[]): string[] => {
  const divisions = ['AL East', 'AL Central', 'AL West', 'NL East', 'NL Central', 'NL West'];
  
  return teams
    .sort((a, b) => a.localeCompare(b))
    .sort((a, b) => {
      const divA = teamDivisions[a.toLowerCase()]?.division;
      const divB = teamDivisions[b.toLowerCase()]?.division;
      return divisions.indexOf(divA) - divisions.indexOf(divB);
    });
};