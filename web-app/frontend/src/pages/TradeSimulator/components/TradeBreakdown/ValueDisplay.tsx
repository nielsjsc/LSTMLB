import React from 'react';
import { TradeAnalysis } from '../../../../services/api';
import TradeMeter from '../TradeMeter/TradeMeter';

interface ValueDisplayProps {
  analysis: TradeAnalysis;
  team1Name: string;
  team2Name: string;
}

const ValueDisplay: React.FC<ValueDisplayProps> = ({ analysis, team1Name, team2Name }) => {
  const { team1, team2 } = analysis;
  const tradeDifferential = team1.total_surplus - team2.total_surplus;

  const formatValue = (value: number | undefined) => {
    if (value === undefined) return '-';
    return `$${(value / 1000000).toFixed(1)}M`;
  };
  
  const renderAssetDetails = (asset: typeof team1.assets[0]) => {
    // Check if asset is a prospect by checking for value property
    if ('value' in asset) {
      return (
        <div className="grid grid-cols-2 gap-2 text-sm">
          <p>FV: {asset.fv || '-'}</p>
          <p>Position: {asset.position || '-'}</p>
          <p className={asset.total_surplus >= 0 ? 'text-green-600' : 'text-red-600'}>
            Value: {formatValue(asset.value)}
          </p>
        </div>
      );
    }
  
    // MLB Player
    return (
      <div className="grid grid-cols-2 gap-2 text-sm">
        <p>Total WAR: {asset.war?.toFixed(1) || '-'}</p>
        <p>Production: {formatValue(asset.total_production)}</p>
        <p>Contract: {formatValue(asset.total_contract)}</p>
        <p className={asset.total_surplus >= 0 ? 'text-green-600' : 'text-red-600'}>
          Trade Value: {formatValue(asset.total_surplus)}
        </p>
      </div>
    );
  };

  return (
    <div className="mt-6 border-t border-slate-700 pt-6">
      <TradeMeter 
        team1Name={team1Name}
        team2Name={team2Name}
        differential={tradeDifferential}
      />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[team1, team2].map((team, index) => (
          <div key={index} className="border border-slate-700 rounded-lg p-4">
            <h3 className="font-bold text-lg mb-4 text-white">
              {index === 0 ? team1Name : team2Name}
            </h3>
            
            <div className="space-y-4">
              {team.assets.map(asset => (
                <div key={asset.name} className="border border-slate-700 p-4 rounded-lg">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-semibold text-lg text-gray-300">{asset.name}</h4>
                  </div>
                  <div className="text-gray-400">
                    {renderAssetDetails(asset)}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 pt-4 border-t border-slate-700">
              <div className="grid grid-cols-2 gap-2 text-gray-300">
                <p>Total Production: {formatValue(team.total_production)}</p>
                <p>Total Contract: {formatValue(team.total_contract)}</p>
                <p className="col-span-2 font-bold text-center text-white">
                  Trade Value: {formatValue(team.total_surplus)}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 text-center p-4 border border-slate-700 rounded-lg">
        <p className="text-lg font-semibold text-gray-300">
          Trade Differential: 
          <span className={`ml-2 ${tradeDifferential >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {formatValue(Math.abs(tradeDifferential))}
            {` in favor of ${tradeDifferential > 0 ? team1Name : team2Name}`}
          </span>
        </p>
      </div>
    </div>
  );
};

export default ValueDisplay;