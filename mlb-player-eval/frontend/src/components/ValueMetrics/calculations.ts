import { Player } from '../../data/mockData'
import { TradeValue, TeamTradeMetrics } from './interfaces'

const WAR_VALUE = 8; // $8M per WAR (industry standard)

export const calculateTradeValue = (players: Player[]): TradeValue => {
  return {
    currentValue: players.reduce((sum, p) => sum + p.value, 0),
    projectedValue: players.reduce((sum, p) => sum + (p.projectedWAR * WAR_VALUE), 0),
    salaryImpact: players.reduce((sum, p) => sum + p.salary, 0),
    warDifferential: players.reduce((sum, p) => sum + (p.projectedWAR - p.war), 0)
  }
}

export const calculateTeamMetrics = (
  team: string, 
  receivingPlayers: Player[]
): TeamTradeMetrics => {
  return {
    team,
    playersReceiving: receivingPlayers,
    totalValue: calculateTradeValue(receivingPlayers)
  }
}