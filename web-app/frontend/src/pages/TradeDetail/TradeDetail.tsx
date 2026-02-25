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

// ── Player card ─────────────────────────────────────────────────────────────

function PlayerCard({
  player,
  teamColor,
}: {
  player: TradePlayerDetail;
  teamColor: string;
}) {
  const surplusPositive = player.surplus >= 0;

  return (
    <div className="rounded-lg border border-white/[0.06] bg-surface-900/50 p-3">
      {/* Player name + link */}
      <div className="flex items-center justify-between mb-2">
        <Link
          to={`/players/${player.mlb_id}`}
          className="text-sm font-semibold text-blue-400 hover:text-blue-300 transition-colors"
        >
          {player.name}
        </Link>
        <div className="flex items-center gap-1.5">
          {player.prospect_fv && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/15">
              FV {player.prospect_fv}
            </span>
          )}
          {player.prospect_top_100 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/15">
              Top 100
            </span>
          )}
          {player.still_on_team && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/15">
              Active
            </span>
          )}
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-2 mb-2">
        <div>
          <div className="text-[10px] text-surface-500 uppercase tracking-wider">WAR</div>
          <div className="text-sm font-semibold text-surface-200">{player.war_with_team}</div>
        </div>
        <div>
          <div className="text-[10px] text-surface-500 uppercase tracking-wider">Seasons</div>
          <div className="text-sm font-semibold text-surface-200">{player.seasons_with_team}</div>
        </div>
        <div>
          <div className="text-[10px] text-surface-500 uppercase tracking-wider">Salary</div>
          <div className="text-sm font-semibold text-surface-200">{fmtMoney(player.salary_with_team, true)}</div>
        </div>
        <div>
          <div className="text-[10px] text-surface-500 uppercase tracking-wider">Surplus</div>
          <div className={`text-sm font-semibold ${surplusPositive ? 'text-emerald-400' : 'text-red-400'}`}>
            {fmtMoney(player.surplus, true)}
          </div>
        </div>
      </div>

      {/* Yearly WAR bar chart */}
      {player.yearly_war.length > 0 && (
        <div className="mt-2">
          <div className="text-[10px] text-surface-500 uppercase tracking-wider mb-1">Yearly WAR</div>
          <div className="flex items-end gap-0.5 h-12">
            {player.yearly_war.map((yw) => {
              const maxWar = Math.max(...player.yearly_war.map(y => Math.abs(y.war)), 1);
              const height = Math.max(2, (Math.abs(yw.war) / maxWar) * 100);
              const isNeg = yw.war < 0;
              return (
                <div key={yw.year} className="flex-1 flex flex-col items-center gap-0.5">
                  <div
                    className="w-full rounded-sm transition-all"
                    style={{
                      height: `${height}%`,
                      backgroundColor: isNeg
                        ? 'rgba(239, 68, 68, 0.4)'
                        : teamColor + '60',
                      minHeight: 2,
                    }}
                    title={`${yw.year}: ${yw.war} WAR`}
                  />
                  <span className="text-[8px] text-surface-600">{String(yw.year).slice(-2)}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Departure info */}
      {!player.still_on_team && player.departure_year && (
        <div className="mt-2 text-[10px] text-surface-500">
          Left after {player.departure_year - 1} season
        </div>
      )}
    </div>
  );
}

// ── Side panel ──────────────────────────────────────────────────────────────

function SidePanel({
  side,
  isWinner,
}: {
  side: TradeSideDetail;
  isWinner: boolean;
}) {
  const colors = getTeamColors(side.team);

  return (
    <div className="flex-1 min-w-0">
      {/* Team header */}
      <div
        className="flex items-center gap-3 mb-4 pb-3 border-b"
        style={{ borderBottomColor: colors.primary + '30' }}
      >
        <div
          className="w-3 h-3 rounded-full flex-shrink-0"
          style={{ backgroundColor: colors.primary }}
        />
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-bold text-surface-100 truncate">
              {side.team_name}
            </h3>
            {isWinner && (
              <span className="text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/20 flex-shrink-0">
                Winner
              </span>
            )}
          </div>
          <p className="text-xs text-surface-500">Received {side.players_received.length} player{side.players_received.length !== 1 ? 's' : ''}</p>
        </div>
      </div>

      {/* Aggregate stats */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="rounded-lg border border-white/[0.06] bg-surface-900/30 p-3 text-center">
          <div className="text-[10px] text-surface-500 uppercase tracking-wider mb-0.5">Total WAR</div>
          <div className="text-xl font-bold text-surface-100">{side.total_war}</div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-surface-900/30 p-3 text-center">
          <div className="text-[10px] text-surface-500 uppercase tracking-wider mb-0.5">Net Surplus</div>
          <div className={`text-xl font-bold ${side.total_surplus >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {fmtMoney(side.total_surplus, true)}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-surface-900/30 p-3 text-center">
          <div className="text-[10px] text-surface-500 uppercase tracking-wider mb-0.5">WAR Value</div>
          <div className="text-sm font-semibold text-surface-200">{fmtMoney(side.total_war_value, true)}</div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-surface-900/30 p-3 text-center">
          <div className="text-[10px] text-surface-500 uppercase tracking-wider mb-0.5">Salary Paid</div>
          <div className="text-sm font-semibold text-surface-200">{fmtMoney(side.total_salary, true)}</div>
        </div>
      </div>

      {/* Player cards */}
      <div className="space-y-2">
        {side.players_received
          .sort((a, b) => b.war_with_team - a.war_with_team)
          .map((p) => (
            <PlayerCard key={p.mlb_id} player={p} teamColor={colors.primary} />
          ))}
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
        <div className="w-6 h-6 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
      </div>
    );
  }

  if (error || !trade) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12 text-center">
        <h2 className="text-xl font-bold text-surface-200 mb-2">Trade Not Found</h2>
        <p className="text-surface-500 mb-6">{error || 'This trade could not be loaded.'}</p>
        <Link to="/trades" className="text-blue-400 hover:text-blue-300 text-sm">
          Back to Past Trades
        </Link>
      </div>
    );
  }

  const winnerColors = getTeamColors(trade.winner);

  // Sort sides: winner first
  const sortedSides = [...trade.sides].sort((a, b) => {
    if (a.team === trade.winner) return -1;
    if (b.team === trade.winner) return 1;
    return 0;
  });

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Back link */}
      <Link
        to="/trades"
        className="inline-flex items-center gap-1 text-sm text-surface-500 hover:text-surface-300 mb-6"
      >
        <span>&larr;</span> Past Trades
      </Link>

      {/* Trade header */}
      <div className="rounded-xl border border-white/[0.06] bg-surface-800/50 p-6 mb-6">
        {/* Date & badges */}
        <div className="flex items-center gap-3 mb-3">
          <span className="text-surface-400 text-sm">{fmtDate(trade.date)}</span>
          {trade.has_cash && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/15">
              Cash Included
            </span>
          )}
          {trade.has_ptbnl && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-700 text-surface-400 border border-white/[0.06]">
              PTBNL
            </span>
          )}
          {trade.n_teams > 2 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/15">
              {trade.n_teams}-Team Trade
            </span>
          )}
        </div>

        {/* Description */}
        <p className="text-surface-300 text-sm leading-relaxed mb-4">
          {trade.description}
        </p>

        {/* Winner announcement */}
        <div
          className="rounded-lg p-4 flex items-center justify-between"
          style={{
            backgroundColor: winnerColors.primary + '08',
            border: `1px solid ${winnerColors.primary}20`,
          }}
        >
          <div>
            <div className="text-xs text-surface-500 uppercase tracking-wider mb-0.5">Trade Winner</div>
            <div className="text-lg font-bold text-surface-100">{trade.winner_name}</div>
          </div>
          <div className="text-right">
            <div className="text-xs text-surface-500 uppercase tracking-wider mb-0.5">Surplus Advantage</div>
            <div className="text-lg font-bold text-emerald-400">+{fmtMoney(trade.surplus_diff, true)}</div>
          </div>
          <div className="text-right">
            <div className="text-xs text-surface-500 uppercase tracking-wider mb-0.5">Total WAR</div>
            <div className="text-lg font-bold text-surface-100">{trade.total_trade_war}</div>
          </div>
        </div>
      </div>

      {/* Side-by-side panels */}
      <div className="flex gap-6">
        {sortedSides.map((side) => (
          <SidePanel
            key={side.team}
            side={side}
            isWinner={side.team === trade.winner}
          />
        ))}
      </div>
    </div>
  );
}
