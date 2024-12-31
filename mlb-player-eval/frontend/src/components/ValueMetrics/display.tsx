import React from 'react'
import { TradeAnalysis } from '../../services/api'

interface ValueDisplayProps {
  analysis: TradeAnalysis;
}

const ValueDisplay: React.FC<ValueDisplayProps> = ({ analysis }) => {
  const { team1, team2 } = analysis;
  const tradeDifferential = team1.total_value - team2.total_value;

  const formatValue = (value: number) => {
    return `$${(value / 1000000).toFixed(1)}M`;
  };

  const formatWar = (war: number) => {
    return war.toFixed(1);
  };

  const calculateProductionValue = (war: number) => {
    return war * 8000000; // $8M per WAR
  };

  return (
    <div className="mt-6 border-t pt-6">
      <h2 className="text-xl font-semibold mb-4">Trade Analysis</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[team1, team2].map((team, index) => (
          <div key={index} className="border rounded-lg p-4">
            <h3 className="font-bold text-lg mb-4">
              {team.players[0]?.team.toUpperCase()}
            </h3>
            
            <div className="space-y-4">
              {team.players.map(player => (
                <div key={player.name} className="border-b pb-4">
                  <h4 className="font-semibold">{player.name}</h4>
                  
                  <div className="mt-2">
                    <p className="font-semibold mb-2">Contract Years:</p>
                    <div className="space-y-3">
                      {player.yearly_projections.map(year => (
                        <div key={year.year} className="bg-gray-50 p-3 rounded">
                          <p className="font-medium text-lg mb-2">{year.year}</p>
                          <div className="space-y-1 text-sm">
                            <p>Projected WAR: {formatWar(year.war)}</p>
                            <p>Production Value: {formatValue(year.base_value)}</p>
                            <p>Contract Cost: {formatValue(year.contract_value)}</p>
                            <p className={year.surplus_value > 0 ? 'text-green-600' : 'text-red-600'}>
                              Surplus Value: {formatValue(year.surplus_value)}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-4 pt-4 border-t">
              <p className="font-bold">
                Total Package Value: {formatValue(team.total_value)}
              </p>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 text-center p-4 bg-gray-50 rounded-lg">
        <p className="text-lg font-semibold">
          Trade Differential: 
          <span className={`ml-2 ${tradeDifferential > 0 ? 'text-green-600' : 'text-red-600'}`}>
            {formatValue(Math.abs(tradeDifferential))}
            {tradeDifferential > 0 
              ? ` in favor of ${team2.players[0]?.team.toUpperCase()}`
              : ` in favor of ${team1.players[0]?.team.toUpperCase()}`}
          </span>
        </p>
      </div>
    </div>
  );
};

export default ValueDisplay;