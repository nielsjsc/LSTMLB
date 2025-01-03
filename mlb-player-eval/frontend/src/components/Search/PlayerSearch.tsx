import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { filterPlayers } from '../../services/api'

const PlayerSearch = () => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<string[]>([])
  const navigate = useNavigate()

  const handleSearch = async (value: string) => {
    setQuery(value)
    if (value.length < 2) {
      setResults([])
      return
    }

    try {
      const players = await filterPlayers({ year: 2025, search: value })
      setResults(players.players.map(p => p.name))
    } catch (err) {
      console.error('Search failed:', err)
    }
  }

  const handleSelect = (playerName: string) => {
    navigate(`/player/${encodeURIComponent(playerName)}`)
    setQuery('')
    setResults([])
  }

  return (
    <div className="relative">
      <input
        type="text"
        value={query}
        onChange={(e) => handleSearch(e.target.value)}
        placeholder="Search players..."
        className="w-full p-2 border rounded text-black bg-white"
      />
      {results.length > 0 && (
        <div className="absolute z-10 w-full mt-1 bg-white border rounded shadow-lg">
          {results.map((name) => (
            <button
              key={name}
              onClick={() => handleSelect(name)}
              className="w-full p-2 text-left text-gray-800 hover:bg-gray-100"
            >
              {name}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default PlayerSearch