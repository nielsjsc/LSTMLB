import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getPastTradeDetail, PastTradeDetail as PastTradeDetailType, TradeSideDetail, TradePlayerDetail } from '../../services/api';

// ── Helpers ──────────────────────────────────────────────────────────────────

const fmtDate = (d: string) => {
  const dt = new Date(d + 'T12:00:00');
  return dt.toLocaleDateString('en-US', { weekday: 'short', month: 'long', day: 'numeric', year: 'numeric' });
};

/**
 * Parse which team received "cash" in a trade.
 */
const parseCashReceiver = (description: string, sides: TradeSideDetail[]): string | null => {
  const cashMatch = description.match(/and cash to ([A-Z][A-Za-z .]+?)(?:\s+for\b|\.$)/i);
  if (!cashMatch) return null;
  const receiverName = cashMatch[1].trim();
  for (const side of sides) {
    if (side.team_name === receiverName || receiverName.startsWith(side.team_name)) {
      return side.team;
    }
  }
  return null;
};

// ── Player card ─────────────────────────────────────────────────────────────

function PlayerCard({ player }: { player: TradePlayerDetail }) {
  const isProspect = !!player.prospect_fv;
  const isPureProspect = isProspect && player.seasons_with_team === 0 && player.war_with_team === 0;
  const noData = player.has_data === false;

  // Link target
  const prospectLink = player.prospect_id ? `/prospects/${player.prospect_id}` : null;
  const mlbLink = !isPureProspect ? `/players/${player.mlb_id}` : null;
  const linkTo = prospectLink || mlbLink;

  // Combine actual + projected WAR into one timeline
  const actualYears = player.yearly_war || [];
  const projectedYears = player.projected_yearly_war || [];

  return (
    <div className="py-3 border-b border-white/[0.04] last:border-b-0">
      {/* Name row */}
      <div className="flex items-center gap-2 mb-1.5 min-w-0">
        {linkTo ? (
          <Link
            to={linkTo}
            className="text-[13px] font-medium text-blue-400 hover:text-blue-300 transition-colors truncate"
          >
            {player.name}
          </Link>
        ) : (
          <span className="text-[13px] font-medium text-surface-200 truncate">
            {player.name}
          </span>
        )}
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
        {player.still_on_team && !isPureProspect && (
          <span className="text-[10px] px-1 py-px rounded bg-emerald-500/10 text-emerald-400/80 flex-shrink-0">
            Active
          </span>
        )}
        {noData && (
          <span className="text-[10px] px-1.5 py-px rounded bg-surface-700/50 text-surface-500 italic flex-shrink-0">
            No Data
          </span>
        )}
      </div>

      {/* Content */}
      {noData ? (
        <div className="text-[11px] text-surface-600 italic">
          No projections, historical stats, or prospect data in our system.
        </div>
      ) : isPureProspect ? (
        <div className="flex items-center gap-4 text-[11px] text-surface-400">
          {player.prospect_rank && (
            <span>
              <span className="text-surface-500">Org Rank</span>{' '}
              <span className="text-surface-200 font-medium">#{player.prospect_rank}</span>
            </span>
          )}
        </div>
      ) : (
        <>
          <div className="flex items-center gap-4 text-[11px] text-surface-400">
            <span>
              <span className="text-surface-500">WAR</span>{' '}
              <span className="text-surface-200 font-medium">{player.war_with_team}</span>
            </span>
            <span>
              <span className="text-surface-500">Seasons</span>{' '}
              <span className="text-surface-200 font-medium">{player.seasons_with_team}</span>
            </span>
          </div>

          {/* Year-by-year WAR: actual + projected */}
          {(actualYears.length > 0 || projectedYears.length > 0) && (
            <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
              {/* Actual years */}
              {actualYears.map((yw) => (
                <span
                  key={yw.year}
                  className={`text-[10px] px-1.5 py-0.5 rounded font-mono tabular-nums ${
                    yw.war >= 3
                      ? 'bg-emerald-500/15 text-emerald-400'
                      : yw.war >= 1
                        ? 'bg-sky-500/10 text-sky-400'
                        : yw.war >= 0
                          ? 'bg-surface-700/60 text-surface-400'
                          : 'bg-red-500/10 text-red-400'
                  }`}
                >
                  {yw.year}: {yw.war > 0 ? '+' : ''}{yw.war}
                </span>
              ))}

              {/* Projected future years (visually distinct) */}
              {projectedYears.length > 0 && actualYears.length > 0 && (
                <span className="text-[9px] text-surface-600 mx-0.5">|</span>
              )}
              {projectedYears.map((yw) => (
                <span
                  key={`proj-${yw.year}`}
                  className="text-[10px] px-1.5 py-0.5 rounded font-mono tabular-nums border border-dashed border-surface-600/50 text-surface-500 bg-surface-800/30"
                  title="Projected"
                >
                  {yw.year}: {yw.war > 0 ? '+' : ''}{yw.war}
                </span>
              ))}
            </div>
          )}

          {/* Departure info */}
          {!player.still_on_team && player.departure_year && (
            <div className="mt-1 text-[10px] text-surface-600">
              Left after {player.departure_year - 1}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Side panel ──────────────────────────────────────────────────────────────

function SidePanel({
  side,
  isWinner,
  receivedCash,
}: {
  side: TradeSideDetail;
  isWinner: boolean;
  receivedCash: boolean;
}) {
  return (
    <div className="flex-1 min-w-0">
      {/* Team header */}
      <div className="flex items-center gap-2 mb-3">
        <h3 className="text-base font-bold text-surface-100 truncate">
          {side.team_name}
        </h3>
        {isWinner && (
          <span className="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 flex-shrink-0">
            Winner
          </span>
        )}
      </div>

      {/* Aggregate: Total WAR */}
      <div className="flex items-center gap-5 mb-3 text-[11px]">
        <div>
          <span className="text-surface-500">Total WAR</span>
          <div className="text-lg font-bold text-surface-100 leading-tight">{side.total_war}</div>
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
            const aIsProspect = !!a.prospect_fv && a.seasons_with_team === 0 && a.war_with_team === 0;
            const bIsProspect = !!b.prospect_fv && b.seasons_with_team === 0 && b.war_with_team === 0;
            if (aIsProspect !== bIsProspect) return aIsProspect ? 1 : -1;
            if (aIsProspect && bIsProspect) {
              return (b.prospect_value ?? 0) - (a.prospect_value ?? 0);
            }
            return b.war_with_team - a.war_with_team;
          })
          .map((p) => (
            <PlayerCard key={p.mlb_id} player={p} />
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
        <div className="w-6 h-6 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
      </div>
    );
  }

  if (error || !trade) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12 text-center">
        <h2 className="text-lg font-bold text-surface-200 mb-2">Trade Not Found</h2>
        <p className="text-surface-500 text-sm mb-4">{error || 'This trade could not be loaded.'}</p>
        <Link to="/trades" className="text-blue-400 hover:text-blue-300 text-sm">
          Back to Trade History
        </Link>
      </div>
    );
  }

  // Determine cash receiver
  const cashReceiver = trade.has_cash
    ? parseCashReceiver(trade.description, trade.sides)
    : null;

  // Sort sides: winner first
  const sortedSides = [...trade.sides].sort((a, b) => {
    if (a.team === trade.winner) return -1;
    if (b.team === trade.winner) return 1;
    return 0;
  });

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      {/* Back link */}
      <Link
        to="/trades"
        className="inline-flex items-center gap-1 text-[12px] text-surface-500 hover:text-surface-300 mb-6 transition-colors"
      >
        <span>←</span> Trade History
      </Link>

      {/* Trade header */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-[13px] text-surface-400">{fmtDate(trade.date)}</span>
          {trade.has_ptbnl && (
            <span className="text-[9px] px-1 py-px rounded bg-surface-700 text-surface-400">PTBNL</span>
          )}
          {trade.n_teams > 2 && (
            <span className="text-[9px] px-1 py-px rounded bg-purple-500/10 text-purple-400">{trade.n_teams}-Team</span>
          )}
        </div>

        <p className="text-[13px] text-surface-300 leading-relaxed mb-6 max-w-3xl">
          {trade.description}
        </p>

        {/* Winner strip */}
        <div
          className="flex items-center justify-between rounded-xl px-5 py-4 bg-surface-800/40 border border-white/[0.06]"
        >
          <div>
            <div className="text-[10px] text-surface-500 uppercase tracking-wider">
              {trade.evaluation_confidence === 'early' ? 'Early Leader' : trade.evaluation_confidence === 'maturing' ? 'Leading' : 'Winner'}
            </div>
            <div className="text-base font-bold text-surface-100">{trade.winner_name}</div>
          </div>
          <div className="text-right">
            <div className="text-[10px] text-surface-500 uppercase tracking-wider">WAR Advantage</div>
            <div className="text-base font-bold text-emerald-400 tabular-nums">
              +{(
                (sortedSides[0]?.total_war ?? 0) - (sortedSides[1]?.total_war ?? 0)
              ).toFixed(1)}
            </div>
          </div>
          <div className="text-right">
            <div className="text-[10px] text-surface-500 uppercase tracking-wider">Total WAR</div>
            <div className="text-base font-bold text-surface-100 tabular-nums">{trade.total_trade_war}</div>
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
              receivedCash={cashReceiver === side.team}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
