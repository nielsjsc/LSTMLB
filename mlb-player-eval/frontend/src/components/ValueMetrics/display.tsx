import React from 'react'
import { TeamTradeMetrics } from './interfaces'

interface ValueDisplayProps {
  metrics: TeamTradeMetrics[];
}

const ValueDisplay: React.FC<ValueDisplayProps> = ({ metrics }) => {
  const getDifferential = () => {
    if (metrics.length !== 2) return 0;
    return metrics[0].totalValue.currentValue - metrics[1].totalValue.currentValue;
  }

  const formatCurrency = (value: number) => `$${value.toFixed(1)}M`;

  return (
    <div className="mt-6 border-t pt-6">
      <h2 className="text-xl font-semibold mb-4">Trade Analysis</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {metrics.map(({ team, totalValue, playersReceiving }) => (
          <div key={team} className="border rounded-lg p-4 shadow-sm">
            <h3 className="font-bold text-lg mb-3">{team} Receives:</h3>
            <div className="space-y-2">
              <p className="flex justify-between">
                <span>Current Value:</span>
                <span className="font-semibold">{formatCurrency(totalValue.currentValue)}</span>
              </p>
              <p className="flex justify-between">
                <span>Projected Value:</span>
                <span className="font-semibold text-blue-600">{formatCurrency(totalValue.projectedValue)}</span>
              </p>
              <p className="flex justify-between">
                <span>Salary Impact:</span>
                <span className="font-semibold text-green-600">{formatCurrency(totalValue.salaryImpact)}</span>
              </p>
              <p className="flex justify-between">
                <span>WAR Differential:</span>
                <span className={`font-semibold ${totalValue.warDifferential > 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {totalValue.warDifferential.toFixed(1)}
                </span>
              </p>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-6 text-center p-4 bg-gray-50 rounded-lg">
        <p className="text-lg font-semibold">
          Trade Differential: 
          <span className={`ml-2 ${getDifferential() > 0 ? 'text-green-600' : 'text-red-600'}`}>
            {formatCurrency(Math.abs(getDifferential()))}
          </span>
        </p>
      </div>
    </div>
  )
}

export default ValueDisplay