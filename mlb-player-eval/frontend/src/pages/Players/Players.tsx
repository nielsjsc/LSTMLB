import React, { useState, useEffect } from 'react'
import { filterPlayers, PlayerResponse, PlayerFilter } from '../../services/api'

const Players = () => {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [players, setPlayers] = useState<PlayerResponse>({ count: 0, players: [] })
  const [teams, setTeams] = useState<string[]>([])
  const [filters, setFilters] = useState<PlayerFilter>({
    year: 2025,
    team: undefined,
    position: undefined,
    sort_by: 'war'
  })

  // Fetch initial data and extract teams
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const data = await filterPlayers({ year: 2025 })
        // Get unique teams and sort alphabetically
        const uniqueTeams = [...new Set(data.players.map(p => p.team.toUpperCase()))].sort()
        setTeams(uniqueTeams)
        setPlayers(data)
      } catch (err) {
        setError('Failed to load initial data')
      } finally {
        setLoading(false)
      }
    }
    fetchInitialData()
  }, [])

  // Handle filter changes
  useEffect(() => {
    const fetchPlayers = async () => {
      if (!filters.year) return // Skip if no year selected
      setLoading(true)
      try {
        const data = await filterPlayers(filters)
        setPlayers(data)
        setError(null)
      } catch (err) {
        setError('Failed to load players')
      } finally {
        setLoading(false)
      }
    }
    fetchPlayers()
  }, [filters])

  const formatValue = (value: number) => `$${(value / 1000000).toFixed(1)}M`
  const formatWar = (war: number) => war.toFixed(1)

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">MLB Players</h1>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      <div className="flex gap-4 mb-6">
        <select
          className="border rounded p-2"
          value={filters.team || ''}
          onChange={(e) => setFilters({ ...filters, team: e.target.value || undefined })}
        >
          <option value="">All Teams</option>
          {teams.map(team => (
            <option key={team} value={team}>{team}</option>
          ))}
        </select>

        <select
          className="border rounded p-2"
          value={filters.position || ''}
          onChange={(e) => setFilters({ ...filters, position: e.target.value || undefined })}
        >
          <option value="">All Positions</option>
          {['SP', 'RP', 'C', '1B', '2B', '3B', 'SS', 'OF'].map(pos => (
            <option key={pos} value={pos}>{pos}</option>
          ))}
        </select>
      </div>

      {loading ? (
        <div className="text-center py-8">Loading players...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {players.players.map((player) => (
            <div key={player.name} className="border rounded-lg p-4">
              <h3 className="font-bold text-lg">{player.name}</h3>
              <p className="text-gray-600">{player.team.toUpperCase()} - {player.position}</p>
              <div className="mt-2 space-y-1">
                <p>WAR: {formatWar(player.war)}</p>
                <p>Base Value: {formatValue(player.base_value)}</p>
                <p>Contract: {formatValue(player.contract_value)}</p>
                <p className={player.surplus_value > 0 ? 'text-green-600' : 'text-red-600'}>
                  Surplus: {formatValue(player.surplus_value)}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default Players