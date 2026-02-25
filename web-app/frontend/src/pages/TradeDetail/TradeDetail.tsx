import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getPastTradeDetail, PastTradeDetail as PastTradeDetailType, TradeSideDetail, TradePlayerDetail } from '../../services/api';
import { getTeamColors } from '../../utils/teamColors';

// ── Helpers ──────────────────────────────────────────────────────────────────

const fmtDate = (d: string) => {
  const dt = new Date(d + 'T12:00:00');
  return dt.toLocaleDateString('en-US', { weekday: 'short', month: 'long', day: 'numeric', year: 'numeric' });
};

const fmtMoney = (n: number, short = false) => {
  const abs = Math.abs(n);
  const sign = n < 0 ? '-' : '';
  if (short) {
    if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(1)}M`;
    if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(0)}K`;
    return `${sign}$${abs}`;
  }
  return `${sign}$${abs.toLocaleString()}`;
};

/**
 * Determine which team received "cash" in a trade by parsing the description.
 * Returns the team abbreviation of the receiving side, or null if not determinable.
 */
const parseCashReceiver = (description: string, sides: TradeSideDetail[]): string | null => {
  // Pattern: "TEAM traded [players] and cash to OTHER_TEAM"
  const cashMatch = description.match(/and cash to ([A-Z][A-Za-z .]+?)(?:\s+for\b|\.$)/i);
  if (!cashMatch) return null;
  const receiverName = cashMatch[1].trim();
  // Match against side team names
  for (const side of sides) {
    if (side.team_name === receiverName || receiverName.startsWith(side.team_name)) {
      return side.team;
    }
  }
  return null;
};

// ── Player card ─────────────────────────────────────────────────────────────

function PlayerCard({
  player,
  isProjected,
}: {
  player: TradePlayerDetail;
  isProjected: boolean;
}) {
  const isProspect = !!player.prospect_fv;
  const isPureProspect = isProspect && player.seasons_with_team === 0 && player.war_with_team === 0;

  const displayWar = isProjected && player.has_projection
    ? (player.projected_war ?? 0)
    : player.war_with_team;
  const displaySurplus = isProjected && player.has_projection
    ? (player.projected_surplus ?? 0)
    : player.surplus;
  const displaySalary = isProjected && player.has_projection
    ? (player.projected_salary ?? 0)
    : player.salary_with_team;
  const surplusPositive = displaySurplus >= 0;

  const yearlyWar = isProjected && player.projected_yearly_war?.length
    ? player.projected_yearly_war
    : player.yearly_war;

  return (
    <div className="py-3 border-b border-white/[0.04] last:border-b-0">
      {/* Name row */}
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2 min-w-0">
          <Link
            to={`/players/${player.mlb_id}`}
            className="text-[13px] font-medium text-blue-400 hover:text-blue-300 transition-colors truncate"
          >
            {player.name}
          </Link>
          {player.prospect_fv && (
            <span className="text-[10px] px-1 py-px rounded bg-amber-500/10 text-amber-400/80 flex-shrink-0">
              FV {player.prospect_fv}
            </span>
          )}
          {player.prospect_top_100 && (
            <span className="text-[10px] px-1 py-px rounded bg-purple-500/10 text-purple-400/80 flex-shrink-0">
              T100
            </span>
          )}
          {player.prospect_level && isPureProspect && (
            <span className="text-[10px] px-1 py-px rounded bg-surface-700 text-surface-400 flex-shrink-0">
              {player.prospect_level}
            </span>
          )}
          {!isProjected && player.still_on_team && !isPureProspect && (
            <span className="text-[10px] px-1 py-px rounded bg-emerald-500/10 text-emerald-400/80 flex-shrink-0">
              Active
            </span>
          )}
          {isProjected && !player.has_projection && !isPureProspect && (
            <span className="text-[10px] text-surface-600 italic flex-shrink-0">
              No projection
            </span>
          )}
        </div>
      </div>

      {/* Inline stats — different layout for pure prospects vs MLB players */}
      {isPureProspect ? (
        <div className="flex items-center gap-4 text-[11px] text-surface-400">
          {player.prospect_rank && (
            <span>
              <span className="text-surface-500">Org Rank</span>{' '}
              <span className="text-surface-200 font-medium">#{player.prospect_rank}</span>
            </span>
          )}
          <span>
            <span className="text-surface-500">$ Value</span>{' '}
            <span className="text-emerald-400 font-medium">
              {fmtMoney(player.prospect_value ?? 0, true)}
            </span>
          </span>
        </div>
      ) : (
        <div className="flex items-center gap-4 text-[11px] text-surface-400">
          <span>
            <span className="text-surface-500">{isProjected ? 'Proj. WAR' : 'WAR'}</span>{' '}
            <span className="text-surface-200 font-medium">{displayWar}</span>
          </span>
          <span>
            <span className="text-surface-500">{isProjected ? 'Yrs' : 'Seasons'}</span>{' '}
            <span className="text-surface-200 font-medium">
              {isProjected ? yearlyWar.length : player.seasons_with_team}
            </span>
          </span>
          <span>
            <span className="text-surface-500">Salary</span>{' '}
            <span className="text-surface-200 font-medium">{fmtMoney(displaySalary, true)}</span>
          </span>
          <span>
            <span className="text-surface-500">Surplus</span>{' '}
            <span className={`font-medium ${surplusPositive ? 'text-emerald-400' : 'text-red-400'}`}>
              {fmtMoney(displaySurplus, true)}
            </span>
          </span>
        </div>
      )}

      {/* Departure info */}
      {!isProjected && !player.still_on_team && player.departure_year && (
        <div className="mt-1 text-[10px] text-surface-600">
          Left after {player.departure_year - 1}
        </div>
      )}
    </div>
  );
}

// ── Side panel ──────────────────────────────────────────────────────────────

function SidePanel({
  side,
  isWinner,
  isProjected,
  receivedCash,
}: {
  side: TradeSideDetail;
  isWinner: boolean;
  isProjected: boolean;
  receivedCash: boolean;
}) {
  const colors = getTeamColors(side.team);

  const displayWar = isProjected ? (side.projected_total_war ?? 0) : side.total_war;
  const displaySurplus = isProjected ? (side.projected_total_surplus ?? 0) : side.total_surplus;
  const displayWarValue = isProjected ? (side.projected_total_war_value ?? 0) : side.total_war_value;
  const displaySalary = isProjected ? (side.projected_total_salary ?? 0) : side.total_salary;

  return (
    <div className="flex-1 min-w-0">
      {/* Team header */}
      <div className="flex items-center gap-3 mb-3">
        <div
          className="w-3 h-3 rounded-full flex-shrink-0"
          style={{ backgroundColor: colors.primary }}
        />
        <h3 className="text-base font-bold text-surface-100 truncate">
          {side.team_name}
        </h3>
        {isWinner && (
          <span className="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 flex-shrink-0">
            Winner
          </span>
        )}
      </div>

      {/* Aggregate stats — horizontal strip */}
      <div className="flex items-center gap-5 mb-3 text-[11px]">
        <div>
          <span className="text-surface-500">{isProjected ? 'Proj. WAR' : 'Total WAR'}</span>
          <div className="text-lg font-bold text-surface-100 leading-tight">{displayWar}</div>
        </div>
        <div>
          <span className="text-surface-500">{isProjected ? 'Proj. Surplus' : 'Net Surplus'}</span>
          <div className={`text-lg font-bold leading-tight ${displaySurplus >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {fmtMoney(displaySurplus, true)}
          </div>
        </div>
        <div>
          <span className="text-surface-500">{isProjected ? 'Value' : 'WAR Value'}</span>
          <div className="text-sm font-semibold text-surface-200 leading-tight">{fmtMoney(displayWarValue, true)}</div>
        </div>
        <div>
          <span className="text-surface-500">{isProjected ? 'Salary' : 'Salary Paid'}</span>
          <div className="text-sm font-semibold text-surface-200 leading-tight">{fmtMoney(displaySalary, true)}</div>
        </div>
      </div>

      {/* Player list */}
      <div>
        <div className="text-[10px] uppercase tracking-wider text-surface-500 mb-1">
          Received {side.players_received.length} player{side.players_received.length !== 1 ? 's' : ''}
          {receivedCash ? ' + cash' : ''}
        </div>
        {side.players_received
          .sort((a, b) => {
            // Sort prospects with value to the end, then by WAR descending
            const aIsProspect = !!a.prospect_fv && a.seasons_with_team === 0 && a.war_with_team === 0;
            const bIsProspect = !!b.prospect_fv && b.seasons_with_team === 0 && b.war_with_team === 0;
            if (aIsProspect !== bIsProspect) return aIsProspect ? 1 : -1;
            if (aIsProspect && bIsProspect) {
              return (b.prospect_value ?? 0) - (a.prospect_value ?? 0);
            }
            const aVal = isProjected && a.has_projection ? (a.projected_war ?? 0) : a.war_with_team;
            const bVal = isProjected && b.has_projection ? (b.projected_war ?? 0) : b.war_with_team;
            return bVal - aVal;
          })
          .map((p) => (
            <PlayerCard key={p.mlb_id} player={p} isProjected={isProjected} />
          ))}
        {receivedCash && (
          <div className="py-2.5 border-b border-white/[0.04] last:border-b-0 flex items-center gap-2">
            <span className="text-[13px] text-amber-400/80 font-medium">Cash considerations</span>
            <span className="text-[10px] px-1 py-px rounded bg-amber-500/10 text-amber-400/70">$</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main page ───────────────────────────────────────────────────────────────

export default function TradeDetail() {
  const { tradeId } = useParams<{ tradeId: string }>();
  const [trade, setTrade] = useState<PastTradeDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!tradeId) return;
    setLoading(true);
    getPastTradeDetail(Number(tradeId))
      .then(setTrade)
      .catch(() => setError('Trade not found'))
      .finally(() => setLoading(false));
  }, [tradeId]);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <div className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
      </div>
    );
  }

  if (error || !trade) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12 text-center">
        <h2 className="text-lg font-bold text-surface-200 mb-2">Trade Not Found</h2>
        <p className="text-surface-500 text-sm mb-4">{error || 'This trade could not be loaded.'}</p>
        <Link to="/trades" className="text-blue-400 hover:text-blue-300 text-sm">
          Back to Past Trades
        </Link>
      </div>
    );
  }

  const winnerColors = getTeamColors(trade.winner);
  const isProjected = trade.evaluation_type === 'projected';
  const displaySurplusDiff = isProjected
    ? (trade.projected_surplus_diff ?? trade.surplus_diff)
    : trade.surplus_diff;
  const displayTotalWar = isProjected
    ? (trade.projected_total_war ?? trade.total_trade_war)
    : trade.total_trade_war;

  // Determine which team received cash (if any)
  const cashReceiver = trade.has_cash
    ? parseCashReceiver(trade.description, trade.sides)
    : null;

  const sortedSides = [...trade.sides].sort((a, b) => {
    if (a.team === trade.winner) return -1;
    if (b.team === trade.winner) return 1;
    return 0;
  });

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Back link */}
      <Link
        to="/trades"
        className="inline-flex items-center gap-1 text-[12px] text-surface-500 hover:text-surface-300 mb-5 transition-colors"
      >
        <span>&larr;</span> Past Trades
      </Link>

      {/* Projected notice */}
      {isProjected && (
        <div className="rounded-lg bg-blue-500/5 border border-blue-500/15 px-4 py-3 mb-4 text-[12px] text-surface-400">
          <span className="text-blue-400 font-medium">Projected evaluation</span> &mdash; too recent for actual data. Values update as players accumulate stats.
        </div>
      )}

      {/* Trade header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-[13px] text-surface-400">{fmtDate(trade.date)}</span>
          {isProjected && (
            <span className="text-[9px] px-1 py-px rounded bg-blue-500/10 text-blue-400 font-medium">Projected</span>
          )}
          {trade.has_ptbnl && (
            <span className="text-[9px] px-1 py-px rounded bg-surface-700 text-surface-400">PTBNL</span>
          )}
          {trade.n_teams > 2 && (
            <span className="text-[9px] px-1 py-px rounded bg-purple-500/10 text-purple-400">{trade.n_teams}-Team</span>
          )}
        </div>

        <p className="text-[13px] text-surface-300 leading-relaxed mb-4 max-w-3xl">
          {trade.description}
        </p>

        {/* Winner strip */}
        <div
          className="flex items-center justify-between rounded-lg px-5 py-3"
          style={{
            backgroundColor: winnerColors.primary + '08',
            borderLeft: `3px solid ${winnerColors.primary}40`,
          }}
        >
          <div>
            <div className="text-[10px] text-surface-500 uppercase tracking-wider">
              {isProjected ? 'Projected Winner' : 'Trade Winner'}
            </div>
            <div className="text-base font-bold text-surface-100">{trade.winner_name}</div>
          </div>
          <div className="text-right">
            <div className="text-[10px] text-surface-500 uppercase tracking-wider">Surplus Adv.</div>
            <div className={`text-base font-bold ${isProjected ? 'text-blue-400' : 'text-emerald-400'}`}>
              +{fmtMoney(displaySurplusDiff, true)}
            </div>
          </div>
          <div className="text-right">
            <div className="text-[10px] text-surface-500 uppercase tracking-wider">Total WAR</div>
            <div className="text-base font-bold text-surface-100">{displayTotalWar}</div>
          </div>
        </div>
      </div>

      {/* Side-by-side panels */}
      <div className="flex gap-8">
        {sortedSides.map((side, idx) => (
          <div key={side.team} className="contents">
            {idx > 0 && (
              <div className="w-px bg-white/[0.06]" />
            )}
            <SidePanel
              side={side}
              isWinner={side.team === trade.winner}
              isProjected={isProjected}
              receivedCash={cashReceiver === side.team}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
