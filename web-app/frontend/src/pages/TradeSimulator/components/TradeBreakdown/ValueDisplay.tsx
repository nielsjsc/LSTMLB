import React from 'react';
import { Link } from 'react-router-dom';
import { TradeAnalysis } from '../../../../services/api';
import { getTeamColors } from '../../../../utils/teamColors';
import { formatCurrency, BALANCED_TRADE_THRESHOLD, getPositionBadgeClasses } from '../../../../utils/tradeValue';
import TradeMeter from '../TradeMeter/TradeMeter';

interface ValueDisplayProps {
  analysis: TradeAnalysis;
  team1Name: string;
  team2Name: string;
}

const TrendingUpIcon: React.FC = () => (
  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 19V5m0 0l-6 6m6-6l6 6" />
  </svg>
);

const TrendingDownIcon: React.FC = () => (
  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 5v14m0 0l-6-6m6 6l6-6" />
  </svg>
);

const ValueDisplay: React.FC<ValueDisplayProps> = ({ analysis, team1Name, team2Name }) => {
  const { team1, team2 } = analysis;
  const tradeDifferential = team1.total_surplus - team2.total_surplus;
  const isBalanced = Math.abs(tradeDifferential) < BALANCED_TRADE_THRESHOLD;

  const team1Colors = getTeamColors(team1Name);
  const team2Colors = getTeamColors(team2Name);

  // Compute max absolute surplus across all assets for bar scaling
  const allAssets = [...team1.assets, ...team2.assets];
  const maxSurplus = Math.max(...allAssets.map(a => Math.abs(a.total_surplus)), 1);

  const renderAssetCard = (asset: typeof team1.assets[0], teamColor: string) => {
    const isProspect = 'value' in asset && !('total_production' in asset);
    const surplusPct = (Math.abs(asset.total_surplus) / maxSurplus) * 100;
    const isPositive = asset.total_surplus >= 0;
    const uniqueKey = 'playerId' in asset ? asset.playerId : ('mlb_id' in asset ? asset.mlb_id : ('real_id' in asset ? asset.real_id : asset.name));

    return (
      <div key={uniqueKey} className="rounded-lg border border-gray-200 bg-white overflow-hidden">
        {/* Asset header */}
        <div className="flex items-center gap-3 px-4 py-3">
          {'position' in asset && (
            <span className={`shrink-0 px-2 py-0.5 rounded text-[10px] font-bold uppercase ${getPositionBadgeClasses(asset.position)}`}>
              {asset.position || '?'}
            </span>
          )}
          <div className="flex-1 min-w-0">
            {'id' in asset && 'war' in asset ? (
              <Link
                to={`/players/${asset.id}`}
                className="font-semibold text-sm text-gray-900 hover:text-brand-500 transition-colors truncate block"
              >
                {asset.name}
              </Link>
            ) : (
              <h4 className="font-semibold text-sm text-gray-900 truncate">{asset.name}</h4>
            )}
            {isProspect && (
              <span className="text-[10px] font-medium text-purple-700 bg-purple-50 border border-purple-200 px-1.5 py-0.5 rounded mt-0.5 inline-block">
                PROSPECT
              </span>
            )}
          </div>
          <span className={`text-sm font-bold tabular-nums ${isPositive ? 'text-emerald-600' : 'text-red-600'}`}>
            {formatCurrency(asset.total_surplus)}
          </span>
        </div>

        {/* Value bar */}
        <div className="px-4 pb-3">
          <div className="h-1.5 rounded-full bg-gray-100 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700 ease-out"
              style={{
                width: `${Math.max(surplusPct, 2)}%`,
                backgroundColor: isPositive ? teamColor : '#dc2626',
              }}
            />
          </div>
        </div>

        {/* Stat breakdown */}
        <div className="px-4 pb-3">
          {isProspect ? (
            <div className="flex items-center gap-4 text-xs text-gray-500">
              {'fv' in asset && (
                <span>FV: <span className="text-gray-700 font-medium">{asset.fv || '—'}</span></span>
              )}
              {'position' in asset && (
                <span>POS: <span className="text-gray-700 font-medium">{asset.position || '—'}</span></span>
              )}
              {'value' in asset && (
                <span>Value: <span className={`font-medium ${(asset.value ?? 0) >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                  {formatCurrency(asset.value)}
                </span></span>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="text-center p-1.5 rounded bg-gray-50">
                <p className="text-gray-400 text-[10px] uppercase tracking-wider mb-0.5">WAR</p>
                <p className="text-gray-800 font-semibold tabular-nums">
                  {'war' in asset ? (asset.war?.toFixed(1) || '—') : '—'}
                </p>
              </div>
              <div className="text-center p-1.5 rounded bg-gray-50">
                <p className="text-gray-400 text-[10px] uppercase tracking-wider mb-0.5">Production</p>
                <p className="text-emerald-600 font-semibold tabular-nums text-[11px]">
                  {formatCurrency(asset.total_production)}
                </p>
              </div>
              <div className="text-center p-1.5 rounded bg-gray-50">
                <p className="text-gray-400 text-[10px] uppercase tracking-wider mb-0.5">Contract</p>
                <p className="text-red-600 font-semibold tabular-nums text-[11px]">
                  {formatCurrency(asset.total_contract)}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderTeamColumn = (
    team: typeof team1,
    teamName: string,
    colors: typeof team1Colors,
    otherTeamTotal: number
  ) => {
    const isAhead = team.total_surplus > otherTeamTotal;
    const isTied = team.total_surplus === otherTeamTotal;

    return (
      <div className="flex flex-col">
        {/* Team header with color accent */}
        <div className="relative rounded-t-xl overflow-hidden">
          <div className="absolute inset-0 opacity-10" style={{ background: colors.gradient }} />
          <div className="relative flex items-center justify-between px-5 py-4">
            <div className="flex items-center gap-3">
              <div className="w-1 h-10 rounded-full" style={{ background: colors.primary }} />
              <div>
                <h3 className="font-bold text-gray-900 text-base tracking-wide">{teamName}</h3>
                <p className="text-xs text-gray-500">{team.assets.length} asset{team.assets.length !== 1 ? 's' : ''}</p>
              </div>
            </div>
            {!isTied && (
              <span className={`inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-full ${
                isAhead
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : 'bg-red-50 text-red-700 border border-red-200'
              }`}>
                {isAhead ? <TrendingUpIcon /> : <TrendingDownIcon />}
                {isAhead ? 'Ahead' : 'Behind'}
              </span>
            )}
          </div>
        </div>

        {/* Assets */}
        <div className="space-y-2 px-1 py-3 flex-1">
          {team.assets.map(asset => renderAssetCard(asset, colors.primary))}
        </div>

        {/* Team totals */}
        <div className="rounded-b-xl border-t border-gray-200 bg-gray-50 px-5 py-4">
          <div className="grid grid-cols-3 gap-3 text-center mb-3">
            <div>
              <p className="text-[10px] text-gray-600 uppercase tracking-wider mb-1 font-semibold">Production</p>
              <p className="text-sm font-bold text-emerald-600 tabular-nums">{formatCurrency(team.total_production)}</p>
            </div>
            <div>
              <p className="text-[10px] text-gray-600 uppercase tracking-wider mb-1 font-semibold">Contract</p>
              <p className="text-sm font-bold text-red-600 tabular-nums">{formatCurrency(team.total_contract)}</p>
            </div>
            <div>
              <p className="text-[10px] text-gray-600 uppercase tracking-wider mb-1 font-semibold">Net Value</p>
              <p className="text-sm font-bold tabular-nums text-gray-900">
                {formatCurrency(team.total_surplus)}
              </p>
            </div>
          </div>
          {/* Visual bar for total */}
          <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-1000 ease-out"
              style={{
                width: `${Math.min((Math.abs(team.total_surplus) / Math.max(Math.abs(team1.total_surplus), Math.abs(team2.total_surplus), 1)) * 100, 100)}%`,
                backgroundColor: colors.primary,
                opacity: 0.85,
              }}
            />
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Trade Meter */}
      <div className="flex justify-center">
        <TradeMeter
          team1Name={team1Name}
          team2Name={team2Name}
          team1Color={team1Colors.primary}
          team2Color={team2Colors.primary}
          differential={tradeDifferential}
        />
      </div>

      {/* Side-by-side comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-xl border border-gray-200 bg-gray-50 overflow-hidden">
          {renderTeamColumn(team1, team1Name, team1Colors, team2.total_surplus)}
        </div>
        <div className="rounded-xl border border-gray-200 bg-gray-50 overflow-hidden">
          {renderTeamColumn(team2, team2Name, team2Colors, team1.total_surplus)}
        </div>
      </div>

      {/* Trade Summary Footer */}
      <div className="relative rounded-xl overflow-hidden border border-gray-200">
        <div className="absolute inset-0 bg-gradient-to-r opacity-[0.03]"
          style={{ backgroundImage: `linear-gradient(90deg, ${team1Colors.primary}, transparent 40%, transparent 60%, ${team2Colors.primary})` }} />
        <div className="relative px-6 py-5 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="w-3 h-3 rounded-full" style={{ background: team1Colors.primary }} />
            <div className="text-center sm:text-left">
              <p className="text-xs text-gray-600 uppercase tracking-wider font-semibold">Trade Differential</p>
              <p className="text-2xl font-bold tabular-nums mt-1 text-gray-900">
                {formatCurrency(Math.abs(tradeDifferential))}
              </p>
            </div>
          </div>
          <div className="text-sm font-medium text-gray-600">
            {isBalanced ? (
              <span className="text-emerald-700 font-semibold">Balanced trade</span>
            ) : (
              <>
                Favors{' '}
                <span className="font-bold text-gray-900">
                  {tradeDifferential > 0 ? team1Name : team2Name}
                </span>
              </>
            )}
          </div>
          <div className="w-3 h-3 rounded-full" style={{ background: team2Colors.primary }} />
        </div>
      </div>
    </div>
  );
};

export default ValueDisplay;