import { Link } from 'react-router-dom';
import { PastTradeDetail, TradeSideDetail } from '../services/api';
import { getTeamColors } from '../utils/teamColors';

// ── Helpers ──────────────────────────────────────────────────────────────────

const fmtDate = (d: string) => {
  const dt = new Date(d + 'T12:00:00');
  return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

const fmtMoney = (n: number) => {
  const abs = Math.abs(n);
  const sign = n < 0 ? '-' : '';
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(0)}K`;
  return `${sign}$${abs}`;
};

// ── Trade card (compact, for player page) ───────────────────────────────────

function PlayerTradeCard({
  trade,
  playerMlbId,
}: {
  trade: PastTradeDetail;
  playerMlbId: number;
}) {
  // Figure out which side this player ended up on
  let playerSide: TradeSideDetail | null = null;
  let otherSide: TradeSideDetail | null = null;

  for (const side of trade.sides) {
    const isOnThisSide = side.players_received.some(p => p.mlb_id === playerMlbId);
    if (isOnThisSide) {
      playerSide = side;
    } else {
      otherSide = side;
    }
  }

  // Find this player's stats
  const playerData = playerSide?.players_received.find(p => p.mlb_id === playerMlbId);

  const isWinner = playerSide?.team === trade.winner;

  return (
    <Link
      to={`/trades/${trade.trade_id}`}
      className="block rounded-lg border border-white/[0.06] bg-surface-900/40 hover:bg-surface-800/60 transition-colors p-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-surface-400">{fmtDate(trade.date)}</span>
          {trade.has_cash && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400/70 border border-amber-500/10">
              + Cash
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {isWinner !== null && (
            <span
              className={`text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded ${
                isWinner
                  ? 'bg-emerald-500/15 text-emerald-400'
                  : 'bg-red-500/10 text-red-400'
              }`}
            >
              {isWinner ? 'W' : 'L'}
            </span>
          )}
          <span className="text-xs text-surface-500">
            {fmtMoney(trade.surplus_diff)} diff
          </span>
        </div>
      </div>

      {/* Trade summary: two sides */}
      <div className="flex gap-4">
        {/* Side that received this player */}
        {playerSide && (
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 mb-1">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: getTeamColors(playerSide.team).primary }}
              />
              <span className="text-xs font-medium text-surface-300">{playerSide.team} received</span>
            </div>
            <div className="space-y-0.5">
              {playerSide.players_received.map(p => (
                <div key={p.mlb_id} className="flex items-center gap-1 text-xs">
                  <span className={p.mlb_id === playerMlbId ? 'text-surface-100 font-medium' : 'text-surface-400'}>
                    {p.name}
                  </span>
                  {p.war_with_team > 0 && (
                    <span className="text-surface-500">{p.war_with_team} WAR</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Divider */}
        <div className="w-px bg-white/[0.06] self-stretch" />

        {/* Other side */}
        {otherSide && (
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 mb-1">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: getTeamColors(otherSide.team).primary }}
              />
              <span className="text-xs font-medium text-surface-300">{otherSide.team} received</span>
            </div>
            <div className="space-y-0.5">
              {otherSide.players_received.map(p => (
                <div key={p.mlb_id} className="flex items-center gap-1 text-xs">
                  <span className="text-surface-400">{p.name}</span>
                  {p.war_with_team > 0 && (
                    <span className="text-surface-500">{p.war_with_team} WAR</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Player's WAR bar */}
      {playerData && playerData.war_with_team > 0 && (
        <div className="mt-3 pt-2 border-t border-white/[0.04]">
          <div className="flex items-center gap-2 text-xs">
            <span className="text-surface-500">Your WAR:</span>
            <span className="text-surface-200 font-semibold">{playerData.war_with_team}</span>
            <span className="text-surface-600">|</span>
            <span className="text-surface-500">Salary:</span>
            <span className="text-surface-200">{fmtMoney(playerData.salary_with_team)}</span>
            <span className="text-surface-600">|</span>
            <span className="text-surface-500">Surplus:</span>
            <span className={playerData.surplus >= 0 ? 'text-emerald-400' : 'text-red-400'}>
              {fmtMoney(playerData.surplus)}
            </span>
          </div>
        </div>
      )}
    </Link>
  );
}

// ── Main component ──────────────────────────────────────────────────────────

export default function PlayerTradeHistory({
  trades,
  playerMlbId,
}: {
  trades: PastTradeDetail[];
  playerMlbId: number;
  teamColor?: string;
}) {
  if (!trades.length) return null;

  return (
    <div>
      <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider mb-3">
        Trade History ({trades.length})
      </h3>
      <div className="space-y-2">
        {trades.map(trade => (
          <PlayerTradeCard key={trade.trade_id} trade={trade} playerMlbId={playerMlbId} />
        ))}
      </div>
    </div>
  );
}
