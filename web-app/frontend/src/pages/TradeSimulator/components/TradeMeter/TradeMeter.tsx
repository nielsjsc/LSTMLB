import React from 'react';
import { formatCurrency, BALANCED_TRADE_THRESHOLD } from '../../../../utils/tradeValue';

interface TradeMeterProps {
  team1Name: string;
  team2Name: string;
  team1Color: string;
  team2Color: string;
  /** team1.total_surplus - team2.total_surplus */
  differential: number;
}

// Differential at which the bar reaches full scale on either side.
const MAX_DIFFERENTIAL = 100_000_000;

const TradeMeter: React.FC<TradeMeterProps> = ({
  team1Name,
  team2Name,
  team1Color,
  team2Color,
  differential,
}) => {
  const absDiff = Math.abs(differential);
  const isBalanced = absDiff < BALANCED_TRADE_THRESHOLD;
  const leadingTeam = differential > 0 ? team1Name : team2Name;
  const leadingColor = differential > 0 ? team1Color : team2Color;
  const fillPct = Math.min((absDiff / MAX_DIFFERENTIAL) * 50, 50);

  return (
    <div className="w-full max-w-md">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-gray-700 truncate">{team1Name}</span>
        <span className="text-xs font-semibold text-gray-700 truncate">{team2Name}</span>
      </div>

      <div className="relative h-3 rounded-full bg-gray-100 overflow-hidden">
        {/* Center (even) marker */}
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-gray-300 z-10" />

        {/* Fill toward whichever team leads */}
        {!isBalanced && (
          <div
            className="absolute top-0 bottom-0 rounded-full transition-all duration-500 ease-out"
            style={{
              left: differential > 0 ? '50%' : `${50 - fillPct}%`,
              width: `${fillPct}%`,
              backgroundColor: leadingColor,
            }}
          />
        )}
      </div>

      <div className="flex justify-between mt-1.5">
        <span className="text-[10px] text-gray-400">{formatCurrency(MAX_DIFFERENTIAL)}</span>
        <span className="text-[10px] text-gray-400">Even</span>
        <span className="text-[10px] text-gray-400">{formatCurrency(MAX_DIFFERENTIAL)}</span>
      </div>

      <p className="text-center text-sm text-gray-600 mt-3">
        {isBalanced ? (
          <span className="text-emerald-700 font-semibold">Balanced trade</span>
        ) : (
          <>
            <span className="font-semibold text-gray-900">{leadingTeam}</span> leads by{' '}
            <span className="font-semibold text-gray-900">{formatCurrency(absDiff)}</span>
          </>
        )}
      </p>
    </div>
  );
};

export default TradeMeter;