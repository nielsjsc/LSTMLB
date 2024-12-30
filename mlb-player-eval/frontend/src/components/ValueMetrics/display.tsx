import React from 'react'
import { TradeAnalysis } from '../../services/api'

interface ValueDisplayProps {
  analysis: TradeAnalysis;
}

const ValueDisplay: React.FC<ValueDisplayProps> = ({ analysis }) => {
  const { team1, team2 } = analysis;
  const tradeDifferential = team1.total_value - team2.total_value;

  const formatValue = (value: number) => {
    return `$${Math.abs(value).toFixed(1)}M`;
  };

  return (
    <div className="mt-6 border-t pt-6">
      <h2 className="text-xl font-semibold mb-4">Trade Analysis</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[team1, team2].map((team, index) => (
          <div key={index} className="border rounded-lg p-4">
            <h3 className="font-bold text-lg mb-4">Team {index + 1}</h3>
            
            <div className="space-y-4">
              {team.players.map(player => (
                <div key={player.name} className="border-b pb-4">
                  <h4 className="font-semibold">{player.name}</h4>
                  <p className="text-sm text-gray-600">{player.team}</p>
                  
                  <div className="mt-2 space-y-1">
                    <p>Total Value: {formatValue(player.total_surplus)}</p>
                    <p>Contract Cost: {formatValue(player.total_contract)}</p>
                    
                    <div className="mt-2">
                      <p className="font-semibold">Yearly Projections:</p>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        {player.yearly_projections.map(year => (
                          <div key={year.year} className="border-l pl-2">
                            <p>{year.year}: {year.war.toFixed(1)} WAR</p>
                            <p className="text-gray-600">
                              {formatValue(year.surplus_value)}
                            </p>
                          </div>
                        ))}
                      </div>
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
            {tradeDifferential > 0 ? ' in favor of Team 1' : ' in favor of Team 2'}
          </span>
        </p>
      </div>
    </div>
  );
};

export default ValueDisplay;