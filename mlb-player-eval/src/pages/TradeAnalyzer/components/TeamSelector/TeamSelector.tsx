import React from 'react'
import { teams } from '../../../../data/mockData'

interface TeamSelectorProps {
  selectedTeams: string[];
  onTeamAdd: (team: string) => void;
  onTeamRemove: (team: string) => void;
}

const TeamSelector = ({ selectedTeams, onTeamAdd, onTeamRemove }: TeamSelectorProps) => {
  return (
    <div className="mb-6">
      <h2 className="text-xl font-semibold mb-4">Select Teams</h2>
      <div className="flex flex-wrap gap-4">
        {selectedTeams.map((team) => (
          <div key={team} className="flex items-center bg-gray-100 rounded-lg p-2">
            <span className="mr-2">{team}</span>
            <button
              onClick={() => onTeamRemove(team)}
              className="text-red-500 hover:text-red-700"
            >
              ×
            </button>
          </div>
        ))}
        {selectedTeams.length < 3 && (
          <select
            onChange={(e) => onTeamAdd(e.target.value)}
            className="border rounded-lg p-2"
            value=""
          >
            <option value="">Add team...</option>
            {teams.filter(t => !selectedTeams.includes(t)).map((team) => (
              <option key={team} value={team}>
                {team}
              </option>
            ))}
          </select>
        )}
      </div>
    </div>
  )
}

export default TeamSelector