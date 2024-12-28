import React from 'react'

const teams = [
  'All', 'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CWS', 'CIN', 'CLE', 'COL', 
  'DET', 'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 
  'OAK', 'PHI', 'PIT', 'SD', 'SF', 'SEA', 'STL', 'TB', 'TEX', 'TOR', 'WSH'
]

interface TeamFilterProps {
  selectedTeam: string;
  onTeamChange: (team: string) => void;
}

const TeamFilter = ({ selectedTeam, onTeamChange }: TeamFilterProps) => {
  return (
    <select 
      value={selectedTeam}
      onChange={(e) => onTeamChange(e.target.value)}
      className="px-4 py-2 border rounded-lg"
    >
      {teams.map((team) => (
        <option key={team} value={team}>
          {team}
        </option>
      ))}
    </select>
  )
}

export default TeamFilter