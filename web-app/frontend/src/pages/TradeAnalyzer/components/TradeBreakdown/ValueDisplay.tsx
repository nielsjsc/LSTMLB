import React from 'react';
import { TradeAnalysis } from '../../../../services/api';
import  TradeMeter  from '../TradeMeter/TradeMeter';

interface ValueDisplayProps {
  analysis: TradeAnalysis;
  team1Name: string;
  team2Name: string;
}

const ValueDisplay: React.FC<ValueDisplayProps> = ({ analysis, team1Name, team2Name }) => {
  const { team1, team2 } = analysis;
  const tradeDifferential = team1.total_surplus - team2.total_surplus;

  const formatValue = (value: number) => `$${(value / 1000000).toFixed(1)}M`;
  
  const getYearRange = (player: typeof team1.players[0]) => {
    const startYear = Math.min(...player.yearly_projections.map(y => y.year));
    const endYear = Math.max(...player.yearly_projections.map(y => y.year));
    return `${startYear}-${endYear}`;
  };

  const getTotalWar = (player: typeof team1.players[0]) => 
    player.yearly_projections.reduce((sum, year) => sum + year.war, 0).toFixed(1);

  return (
    <div className="mt-6 border-t pt-6">
      <TradeMeter 
        team1Name={team1Name}
        team2Name={team2Name}
        differential={tradeDifferential}
      />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[team1, team2].map((team, index) => (
          <div key={index} className="border rounded-lg p-4">
            <h3 className="font-bold text-lg mb-4">{index === 0 ? team1Name : team2Name}</h3>
            
            <div className="space-y-4">
              {team.players.map(player => (
                <div key={player.name} className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-semibold text-lg">{player.name}</h4>
                    <span className="text-sm text-gray-600">{getYearRange(player)}</span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <p>Total WAR: {getTotalWar(player)}</p>
                    <p>Production: {formatValue(player.total_production)}</p>
                    <p>Contract: {formatValue(player.total_contract)}</p>
                    <p className={player.total_surplus >= 0 ? 'text-green-600' : 'text-red-600'}>
                      Surplus: {formatValue(player.total_surplus)}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 pt-4 border-t">
              <div className="grid grid-cols-2 gap-2">
                <p>Total Production: {formatValue(team.total_production)}</p>
                <p>Total Contract: {formatValue(team.total_contract)}</p>
                <p className="col-span-2 font-bold text-center">
                  Total Surplus: {formatValue(team.total_surplus)}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 text-center p-4 bg-gray-50 rounded-lg">
        <p className="text-lg font-semibold">
          Trade Differential: 
          <span className={`ml-2 ${tradeDifferential >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {formatValue(Math.abs(tradeDifferential))}
            {` in favor of ${tradeDifferential > 0 ? team1Name : team2Name}`}
          </span>
        </p>
      </div>
    </div>
  );
};

export default ValueDisplay;