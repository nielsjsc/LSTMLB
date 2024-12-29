import React from 'react'

const positions = [
  'All', 'SP', 'RP', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH'
]

interface PositionFilterProps {
  selectedPosition: string;
  onPositionChange: (position: string) => void;
}

const PositionFilter = ({ selectedPosition, onPositionChange }: PositionFilterProps) => {
  return (
    <select 
      value={selectedPosition}
      onChange={(e) => onPositionChange(e.target.value)}
      className="px-4 py-2 border rounded-lg"
    >
      {positions.map((position) => (
        <option key={position} value={position}>
          {position}
        </option>
      ))}
    </select>
  )
}

export default PositionFilter