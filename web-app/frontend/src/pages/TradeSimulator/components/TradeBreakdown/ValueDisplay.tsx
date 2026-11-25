import React from 'react';
import { Link } from 'react-router-dom';
import { TradeAnalysis, TradeAsset, TradeSideAnalysis } from '../../../../services/api';
import { CURRENT_YEAR } from '../../../../config';
import { getTeamColors } from '../../../../utils/teamColors';
import { formatCurrency, formatRange, getPositionBadgeClasses } from '../../../../utils/tradeValue';
import TradeMeter from '../TradeMeter/TradeMeter';
import ControlYears from '../ControlYears/ControlYears';

interface ValueDisplayProps {
  analysis: TradeAnalysis;
  team1Name: string;
  team2Name: string;
}

const ValueDisplay: React.FC<ValueDisplayProps> = ({ analysis, team1Name, team2Name }) => {
  const { side_a, side_b } = analysis;
  const tradeDifferential = side_a.consolidated_surplus - side_b.consolidated_surplus;

  const team1Colors = getTeamColors(team1Name);
  const team2Colors = getTeamColors(team2Name);

  const renderRow = (asset: TradeAsset) => {
    const isProspect = asset.type === 'prospect';
    const isPositive = asset.trade_value >= 0;
    const linkTarget = asset.mlbam_id || asset.id;
    const rangeLabel = formatRange(asset.trade_value_p10, asset.trade_value_p90);

    return (
      <tr key={`${asset.type}-${linkTarget}-${asset.name}`} className="border-b border-gray-100 last:border-0">
        <td className="px-4 py-3 align-top">
          <div className="flex items-start gap-2.5">
            {asset.position && (
              <span
                className={`shrink-0 mt-0.5 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${getPositionBadgeClasses(
                  asset.position
                )}`}
              >
                {asset.position}
              </span>
            )}
            <div className="min-w-0">
              {linkTarget ? (
                <Link
                  to={`/players/${linkTarget}`}
                  className="text-sm font-medium text-gray-900 hover:text-brand-500 transition-colors truncate block"
                >
                  {asset.name}
                </Link>
              ) : (
                <p className="text-sm font-medium text-gray-900 truncate">{asset.name}</p>
              )}
              <p className="text-xs text-gray-400 mt-0.5">
                {isProspect
                  ? `Prospect${asset.fv ? ` \u00b7 FV ${asset.fv}` : ''}${asset.level ? ` \u00b7 ${asset.level}` : ''}`
                  : `${asset.control_war.toFixed(1)} WAR`}
              </p>
            </div>
          </div>
        </td>
        <td className="px-2 py-3 align-top">
          <ControlYears
            currentYear={CURRENT_YEAR}
            yearsControl={asset.years_control}
            probableFaYear={asset.probable_fa_year}
            faYear={asset.fa_year}
            projections={asset.projections ?? []}
          />
        </td>
        <td className="px-2 py-3 text-right align-top tabular-nums text-sm text-gray-700">
          {!isProspect ? formatCurrency(asset.control_production_value) : '—'}
        </td>
        <td className="px-2 py-3 text-right align-top tabular-nums text-sm text-gray-700">
          {!isProspect ? formatCurrency(asset.control_salary_cost) : '—'}
        </td>
        <td className="px-4 py-3 text-right align-top tabular-nums">
          <div className={`text-sm font-semibold ${isPositive ? 'text-emerald-700' : 'text-red-600'}`}>
            {formatCurrency(asset.trade_value)}
          </div>
          {rangeLabel && (
            <div className="text-[10px] text-gray-400 mt-0.5 whitespace-nowrap" title="10th – 90th percentile surplus">
              {rangeLabel}
            </div>
          )}
        </td>
      </tr>
    );
  };

  const renderTeamTable = (side: TradeSideAnalysis, teamName: string, colors: typeof team1Colors) => {
    const hasConsolidationAdj = Math.abs(side.consolidation_penalty) >= 50_000;
    const sideRange = formatRange(side.surplus_p10, side.surplus_p90);

    return (
      <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center gap-2.5">
            <span className="w-2 h-2 rounded-full shrink-0" style={{ background: colors.primary }} />
            <h3 className="text-sm font-semibold text-gray-900">{teamName} Receives</h3>
          </div>
          <span className="text-xs text-gray-400">
            {side.assets.length} asset{side.assets.length !== 1 ? 's' : ''}
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
                <th className="text-right font-semibold px-4 py-2">Net (P10–P90)</th>
              </tr>
            </thead>
            <tbody>{side.assets.map(renderRow)}</tbody>
            <tfoot>
              {hasConsolidationAdj && (
                <tr className="bg-amber-50/40 border-t border-gray-100 text-xs text-gray-500">
                  <td colSpan={4} className="px-4 py-2 font-medium">
                    Roster Consolidation Adjustment (Raw Sum: {formatCurrency(side.raw_surplus_sum)})
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums font-medium text-amber-700">
                    {formatCurrency(side.consolidation_penalty)}
                  </td>
                </tr>
              )}
              <tr className="bg-gray-50 border-t border-gray-200">
                <td colSpan={2} className="px-4 py-3 text-xs font-semibold uppercase tracking-widest text-gray-500">
                  {hasConsolidationAdj ? 'Consolidated Value' : 'Total'}
                </td>
                <td className="px-2 py-3 text-right tabular-nums text-sm font-semibold text-gray-700">
                  {formatCurrency(side.control_production_value)}
                </td>
                <td className="px-2 py-3 text-right tabular-nums text-sm font-semibold text-gray-700">
                  {formatCurrency(side.control_salary_cost)}
                </td>
                <td className="px-4 py-3 text-right tabular-nums">
                  <div className="text-sm font-bold text-gray-900">
                    {formatCurrency(side.consolidated_surplus)}
                  </div>
                  {sideRange && side.assets.length > 0 && (
                    <div className="text-[10px] text-gray-400 mt-0.5 whitespace-nowrap">
                      {sideRange}
                    </div>
                  )}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-center">
        <TradeMeter
          team1Name={team1Name}
          team2Name={team2Name}
          team1Color={team1Colors.primary}
          team2Color={team2Colors.primary}
          differential={tradeDifferential}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {renderTeamTable(side_a, team1Name, team1Colors)}
        {renderTeamTable(side_b, team2Name, team2Colors)}
      </div>

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
