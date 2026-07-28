import React from 'react';
import { Link } from 'react-router-dom';
import { TradeAnalysis } from '../../../../services/api';
import { CURRENT_YEAR } from '../../../../config';
import { getTeamColors } from '../../../../utils/teamColors';
import { formatCurrency, getPositionBadgeClasses } from '../../../../utils/tradeValue';
import TradeMeter from '../TradeMeter/TradeMeter';
import ControlYears from '../ControlYears/ControlYears';

interface ValueDisplayProps {
  analysis: TradeAnalysis;
  team1Name: string;
  team2Name: string;
}

type TeamAsset = TradeAnalysis['team1']['assets'][number];

/**
 * Contract-control fields aren't part of the TradeAnalysis asset type yet.
 * Add years_control / probable_fa_year / fa_year / projections to the
 * analyzeTrade response and the Player type (mirroring what the player
 * details page already returns) to populate the Control column below.
 * Until then it renders as "—" for any asset missing this data.
 */
type ControlFields = {
  years_control?: number | null;
  probable_fa_year?: number | null;
  fa_year?: number | null;
  projections?: Array<{ year: number; status: string }>;
};

function withControlFields(asset: TeamAsset): TeamAsset & ControlFields {
  return asset;
}

const isProspectAsset = (asset: TeamAsset) => 'value' in asset && !('total_production' in asset);

const assetKey = (asset: TeamAsset) =>
  'playerId' in asset ? asset.playerId : 'mlb_id' in asset ? asset.mlb_id : 'real_id' in asset ? asset.real_id : asset.name;

const ValueDisplay: React.FC<ValueDisplayProps> = ({ analysis, team1Name, team2Name }) => {
  const { team1, team2 } = analysis;
  const tradeDifferential = team1.total_surplus - team2.total_surplus;

  const team1Colors = getTeamColors(team1Name);
  const team2Colors = getTeamColors(team2Name);

  const renderRow = (asset: TeamAsset) => {
    const control = withControlFields(asset);
    const isProspect = isProspectAsset(asset);
    const isPositive = asset.total_surplus >= 0;

    return (
      <tr key={assetKey(asset)} className="border-b border-gray-100 last:border-0">
        <td className="px-4 py-3 align-top">
          <div className="flex items-start gap-2.5">
            {'position' in asset && (
              <span
                className={`shrink-0 mt-0.5 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${getPositionBadgeClasses(
                  asset.position
                )}`}
              >
                {asset.position || '?'}
              </span>
            )}
            <div className="min-w-0">
              {'id' in asset && 'war' in asset ? (
                <Link
                  to={`/players/${asset.id}`}
                  className="text-sm font-medium text-gray-900 hover:text-brand-500 transition-colors truncate block"
                >
                  {asset.name}
                </Link>
              ) : (
                <p className="text-sm font-medium text-gray-900 truncate">{asset.name}</p>
              )}
              <p className="text-xs text-gray-400 mt-0.5">
                {isProspect
                  ? `Prospect${'fv' in asset && asset.fv ? ` \u00b7 FV ${asset.fv}` : ''}`
                  : 'war' in asset && asset.war != null
                  ? `${asset.war.toFixed(1)} WAR`
                  : null}
              </p>
            </div>
          </div>
        </td>
        <td className="px-2 py-3 align-top">
          <ControlYears
            currentYear={CURRENT_YEAR}
            yearsControl={control.years_control}
            probableFaYear={control.probable_fa_year}
            faYear={control.fa_year}
            projections={control.projections}
          />
        </td>
        <td className="px-2 py-3 text-right align-top tabular-nums text-sm text-gray-700">
          {'total_production' in asset ? formatCurrency(asset.total_production) : '—'}
        </td>
        <td className="px-2 py-3 text-right align-top tabular-nums text-sm text-gray-700">
          {'total_contract' in asset ? formatCurrency(asset.total_contract) : '—'}
        </td>
        <td
          className={`px-4 py-3 text-right align-top tabular-nums text-sm font-semibold ${
            isPositive ? 'text-emerald-700' : 'text-red-600'
          }`}
        >
          {formatCurrency(asset.total_surplus)}
        </td>
      </tr>
    );
  };

  const renderTeamTable = (team: typeof team1, teamName: string, colors: typeof team1Colors) => (
    <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center gap-2.5">
          <span className="w-2 h-2 rounded-full shrink-0" style={{ background: colors.primary }} />
          <h3 className="text-sm font-semibold text-gray-900">{teamName}</h3>
        </div>
        <span className="text-xs text-gray-400">
          {team.assets.length} asset{team.assets.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-[10px] uppercase tracking-widest text-gray-400 border-b border-gray-200">
              <th className="text-left font-semibold px-4 py-2">Player</th>
              <th className="text-left font-semibold px-2 py-2">Control</th>
              <th className="text-right font-semibold px-2 py-2">Production</th>
              <th className="text-right font-semibold px-2 py-2">Contract</th>
              <th className="text-right font-semibold px-4 py-2">Net</th>
            </tr>
          </thead>
          <tbody>{team.assets.map(renderRow)}</tbody>
          <tfoot>
            <tr className="bg-gray-50 border-t border-gray-200">
              <td colSpan={2} className="px-4 py-3 text-xs font-semibold uppercase tracking-widest text-gray-500">
                Total
              </td>
              <td className="px-2 py-3 text-right tabular-nums text-sm font-semibold text-gray-700">
                {formatCurrency(team.total_production)}
              </td>
              <td className="px-2 py-3 text-right tabular-nums text-sm font-semibold text-gray-700">
                {formatCurrency(team.total_contract)}
              </td>
              <td className="px-4 py-3 text-right tabular-nums text-sm font-bold text-gray-900">
                {formatCurrency(team.total_surplus)}
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Single source of truth for who leads the trade and by how much */}
      <div className="flex justify-center">
        <TradeMeter
          team1Name={team1Name}
          team2Name={team2Name}
          team1Color={team1Colors.primary}
          team2Color={team2Colors.primary}
          differential={tradeDifferential}
        />
      </div>

      {/* Per-team asset tables — team totals live only in the footer row below */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {renderTeamTable(team1, team1Name, team1Colors)}
        {renderTeamTable(team2, team2Name, team2Colors)}
      </div>

      {/* Control-column legend, shown once for both tables */}
      <div className="flex flex-wrap items-center gap-4 px-1">
        <span className="text-[10px] uppercase tracking-widest text-gray-400">Control</span>
        <Legend swatchClass="bg-gray-700 border-gray-700" label="Under control" />
        <Legend swatchClass="bg-transparent border-dashed border-amber-400" label="Option year" />
        <Legend swatchClass="bg-amber-100 border-amber-300" label="Probable FA" />
        <Legend swatchClass="bg-gray-50 border-gray-200" label="Uncontracted" />
      </div>
    </div>
  );
};

const Legend: React.FC<{ swatchClass: string; label: string }> = ({ swatchClass, label }) => (
  <div className="flex items-center gap-1.5">
    <div className={`w-2.5 h-2.5 rounded-sm border ${swatchClass}`} />
    <span className="text-[10px] text-gray-500">{label}</span>
  </div>
);

export default ValueDisplay;