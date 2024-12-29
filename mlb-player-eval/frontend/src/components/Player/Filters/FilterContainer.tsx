import React from 'react'
import PositionFilter from './PositionFilter'
import TeamFilter from './TeamFilter'

interface FilterContainerProps {
  selectedPosition: string;
  selectedTeam: string;
  onPositionChange: (position: string) => void;
  onTeamChange: (team: string) => void;
}

const FilterContainer = ({ 
  selectedPosition, 
  selectedTeam, 
  onPositionChange, 
  onTeamChange 
}: FilterContainerProps) => {
  return (
    <div className="flex gap-4 mb-6">
      <div>
        <label className="block text-sm font-medium mb-2">Position</label>
        <PositionFilter 
          selectedPosition={selectedPosition} 
          onPositionChange={onPositionChange} 
        />
      </div>
      <div>
        <label className="block text-sm font-medium mb-2">Team</label>
        <TeamFilter 
          selectedTeam={selectedTeam} 
          onTeamChange={onTeamChange} 
        />
      </div>
    </div>
  )
}

export default FilterContainer