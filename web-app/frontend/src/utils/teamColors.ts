/**
 * MLB team color mappings for dynamic theming.
 * Each team has a primary color, secondary color, and accent.
 */

export interface TeamColors {
  primary: string;     // Main team color
  secondary: string;   // Secondary/alternate color
  accent: string;      // Light accent for text highlights
  gradient: string;    // CSS gradient string for headers
  bg: string;          // Subtle background tint (very low opacity)
}

const teamColorMap: Record<string, TeamColors> = {
  // AL East
  BAL: {
    primary: '#DF4601',
    secondary: '#27251F',
    accent: '#F9893D',
    gradient: 'linear-gradient(135deg, #DF4601 0%, #27251F 100%)',
    bg: 'rgba(223, 70, 1, 0.08)',
  },
  BOS: {
    primary: '#BD3039',
    secondary: '#0C2340',
    accent: '#E88A8F',
    gradient: 'linear-gradient(135deg, #BD3039 0%, #0C2340 100%)',
    bg: 'rgba(189, 48, 57, 0.08)',
  },
  NYY: {
    primary: '#003087',
    secondary: '#132448',
    accent: '#6B8FCC',
    gradient: 'linear-gradient(135deg, #003087 0%, #132448 100%)',
    bg: 'rgba(0, 48, 135, 0.08)',
  },
  TBR: {
    primary: '#092C5C',
    secondary: '#8FBCE6',
    accent: '#8FBCE6',
    gradient: 'linear-gradient(135deg, #092C5C 0%, #8FBCE6 100%)',
    bg: 'rgba(9, 44, 92, 0.08)',
  },
  TOR: {
    primary: '#134A8E',
    secondary: '#1D2D5C',
    accent: '#6BA4E8',
    gradient: 'linear-gradient(135deg, #134A8E 0%, #1D2D5C 100%)',
    bg: 'rgba(19, 74, 142, 0.08)',
  },

  // AL Central
  CHW: {
    primary: '#27251F',
    secondary: '#C4CED4',
    accent: '#A0A8AD',
    gradient: 'linear-gradient(135deg, #27251F 0%, #4A4A4A 100%)',
    bg: 'rgba(39, 37, 31, 0.08)',
  },
  CLE: {
    primary: '#00385D',
    secondary: '#E31937',
    accent: '#5B9BD5',
    gradient: 'linear-gradient(135deg, #00385D 0%, #E31937 100%)',
    bg: 'rgba(0, 56, 93, 0.08)',
  },
  DET: {
    primary: '#0C2340',
    secondary: '#FA4616',
    accent: '#5B7FA6',
    gradient: 'linear-gradient(135deg, #0C2340 0%, #FA4616 100%)',
    bg: 'rgba(12, 35, 64, 0.08)',
  },
  KCR: {
    primary: '#004687',
    secondary: '#BD9B60',
    accent: '#6BA4E8',
    gradient: 'linear-gradient(135deg, #004687 0%, #BD9B60 100%)',
    bg: 'rgba(0, 70, 135, 0.08)',
  },
  MIN: {
    primary: '#002B5C',
    secondary: '#D31145',
    accent: '#5B7FA6',
    gradient: 'linear-gradient(135deg, #002B5C 0%, #D31145 100%)',
    bg: 'rgba(0, 43, 92, 0.08)',
  },

  // AL West
  HOU: {
    primary: '#002D62',
    secondary: '#EB6E1F',
    accent: '#F9A64A',
    gradient: 'linear-gradient(135deg, #002D62 0%, #EB6E1F 100%)',
    bg: 'rgba(0, 45, 98, 0.08)',
  },
  LAA: {
    primary: '#BA0021',
    secondary: '#003263',
    accent: '#E85B6F',
    gradient: 'linear-gradient(135deg, #BA0021 0%, #003263 100%)',
    bg: 'rgba(186, 0, 33, 0.08)',
  },
  OAK: {
    primary: '#003831',
    secondary: '#EFB21E',
    accent: '#4DAB8E',
    gradient: 'linear-gradient(135deg, #003831 0%, #EFB21E 100%)',
    bg: 'rgba(0, 56, 49, 0.08)',
  },
  SEA: {
    primary: '#0C2C56',
    secondary: '#005C5C',
    accent: '#4D9999',
    gradient: 'linear-gradient(135deg, #0C2C56 0%, #005C5C 100%)',
    bg: 'rgba(12, 44, 86, 0.08)',
  },
  TEX: {
    primary: '#003278',
    secondary: '#C0111F',
    accent: '#5B8FCC',
    gradient: 'linear-gradient(135deg, #003278 0%, #C0111F 100%)',
    bg: 'rgba(0, 50, 120, 0.08)',
  },

  // NL East
  ATL: {
    primary: '#CE1141',
    secondary: '#13274F',
    accent: '#E86B8A',
    gradient: 'linear-gradient(135deg, #CE1141 0%, #13274F 100%)',
    bg: 'rgba(206, 17, 65, 0.08)',
  },
  MIA: {
    primary: '#00A3E0',
    secondary: '#EF3340',
    accent: '#5BC8F0',
    gradient: 'linear-gradient(135deg, #00A3E0 0%, #EF3340 100%)',
    bg: 'rgba(0, 163, 224, 0.08)',
  },
  NYM: {
    primary: '#002D72',
    secondary: '#FF5910',
    accent: '#5B8FCC',
    gradient: 'linear-gradient(135deg, #002D72 0%, #FF5910 100%)',
    bg: 'rgba(0, 45, 114, 0.08)',
  },
  PHI: {
    primary: '#E81828',
    secondary: '#002D72',
    accent: '#F07078',
    gradient: 'linear-gradient(135deg, #E81828 0%, #002D72 100%)',
    bg: 'rgba(232, 24, 40, 0.08)',
  },
  WSN: {
    primary: '#AB0003',
    secondary: '#14225A',
    accent: '#D45B5D',
    gradient: 'linear-gradient(135deg, #AB0003 0%, #14225A 100%)',
    bg: 'rgba(171, 0, 3, 0.08)',
  },

  // NL Central
  CHC: {
    primary: '#0E3386',
    secondary: '#CC3433',
    accent: '#5B7FCC',
    gradient: 'linear-gradient(135deg, #0E3386 0%, #CC3433 100%)',
    bg: 'rgba(14, 51, 134, 0.08)',
  },
  CIN: {
    primary: '#C6011F',
    secondary: '#000000',
    accent: '#E86B78',
    gradient: 'linear-gradient(135deg, #C6011F 0%, #000000 100%)',
    bg: 'rgba(198, 1, 31, 0.08)',
  },
  MIL: {
    primary: '#FFC52F',
    secondary: '#12284B',
    accent: '#FFD76B',
    gradient: 'linear-gradient(135deg, #12284B 0%, #FFC52F 100%)',
    bg: 'rgba(255, 197, 47, 0.06)',
  },
  PIT: {
    primary: '#27251F',
    secondary: '#FDB827',
    accent: '#FDCF6B',
    gradient: 'linear-gradient(135deg, #27251F 0%, #FDB827 100%)',
    bg: 'rgba(253, 184, 39, 0.06)',
  },
  STL: {
    primary: '#C41E3A',
    secondary: '#0C2340',
    accent: '#E87088',
    gradient: 'linear-gradient(135deg, #C41E3A 0%, #0C2340 100%)',
    bg: 'rgba(196, 30, 58, 0.08)',
  },

  // NL West
  ARI: {
    primary: '#A71930',
    secondary: '#E3D4AD',
    accent: '#D46B78',
    gradient: 'linear-gradient(135deg, #A71930 0%, #E3D4AD 100%)',
    bg: 'rgba(167, 25, 48, 0.08)',
  },
  COL: {
    primary: '#33006F',
    secondary: '#C4CED4',
    accent: '#8040BF',
    gradient: 'linear-gradient(135deg, #33006F 0%, #C4CED4 100%)',
    bg: 'rgba(51, 0, 111, 0.08)',
  },
  LAD: {
    primary: '#005A9C',
    secondary: '#EF3E42',
    accent: '#5BA4E8',
    gradient: 'linear-gradient(135deg, #005A9C 0%, #002F6C 100%)',
    bg: 'rgba(0, 90, 156, 0.08)',
  },
  SDP: {
    primary: '#2F241D',
    secondary: '#FFC425',
    accent: '#FFCF5B',
    gradient: 'linear-gradient(135deg, #2F241D 0%, #FFC425 100%)',
    bg: 'rgba(255, 196, 37, 0.06)',
  },
  SFG: {
    primary: '#FD5A1E',
    secondary: '#27251F',
    accent: '#FD8A5E',
    gradient: 'linear-gradient(135deg, #FD5A1E 0%, #27251F 100%)',
    bg: 'rgba(253, 90, 30, 0.08)',
  },
};

// Default colors for unknown teams or FA
const defaultColors: TeamColors = {
  primary: '#34d399',
  secondary: '#0f172a',
  accent: '#34d399',
  gradient: 'linear-gradient(135deg, #34d399 0%, #60a5fa 100%)',
  bg: 'rgba(52, 211, 153, 0.06)',
};

/**
 * Get team colors for a given team abbreviation.
 * Falls back to brand default colors for unknown teams.
 */
export function getTeamColors(team: string | undefined | null): TeamColors {
  if (!team) return defaultColors;
  return teamColorMap[team.toUpperCase()] || defaultColors;
}

/**
 * Get full team name from abbreviation.
 */
export function getTeamName(abbrev: string): string {
  const names: Record<string, string> = {
    ARI: 'Arizona Diamondbacks', ATL: 'Atlanta Braves', BAL: 'Baltimore Orioles',
    BOS: 'Boston Red Sox', CHC: 'Chicago Cubs', CHW: 'Chicago White Sox',
    CIN: 'Cincinnati Reds', CLE: 'Cleveland Guardians', COL: 'Colorado Rockies',
    DET: 'Detroit Tigers', HOU: 'Houston Astros', KCR: 'Kansas City Royals',
    LAA: 'Los Angeles Angels', LAD: 'Los Angeles Dodgers', MIA: 'Miami Marlins',
    MIL: 'Milwaukee Brewers', MIN: 'Minnesota Twins', NYM: 'New York Mets',
    NYY: 'New York Yankees', OAK: 'Oakland Athletics', PHI: 'Philadelphia Phillies',
    PIT: 'Pittsburgh Pirates', SDP: 'San Diego Padres', SFG: 'San Francisco Giants',
    SEA: 'Seattle Mariners', STL: 'St. Louis Cardinals', TBR: 'Tampa Bay Rays',
    TEX: 'Texas Rangers', TOR: 'Toronto Blue Jays', WSN: 'Washington Nationals',
    FA: 'Free Agent',
  };
  return names[abbrev.toUpperCase()] || abbrev;
}
