import { Player } from '../../data/mockData'

export interface TradeValue {
  currentValue: number;
  projectedValue: number;
  salaryImpact: number;
  warDifferential: number;
}

export interface TeamTradeMetrics {
  team: string;
  playersReceiving: Player[];
  totalValue: TradeValue;
}