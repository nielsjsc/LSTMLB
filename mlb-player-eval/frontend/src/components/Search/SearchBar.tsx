import React from 'react'

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
}

const SearchBar = ({ value, onChange }: SearchBarProps) => {
  return (
    <div className="w-full max-w-xl mx-auto my-4">
      <input
        type="text"
        placeholder="Search players..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-4 py-2 border rounded-lg"
      />
    </div>
  )
}

export default SearchBar