import React, { useState, useEffect, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { getPlayerDetails, getTradeValueHistory, getPlayerTransactions, getPlayerInfo, getPlayerPastTrades, PlayerStats } from '../../services/api';
import type { TradeValuePoint, Transaction, PlayerInfo, PastTradeDetail } from '../../services/api';
import { CURRENT_YEAR, MAX_PROJECTION_YEARS, API_BASE } from '../../config';
import { CombinedHittingTable, CombinedPitchingTable } from '../../components/Tables';
import { getTeamColors, getTeamName } from '../../utils/teamColors';
import TradeValueChart from '../../components/TradeValueChart';
import TransactionHistory from '../../components/TransactionHistory';
import PlayerBioSection from '../../components/PlayerBioSection';
import PlayerTradeHistory from '../../components/PlayerTradeHistory';

// ─── Helpers ────────────────────────────────────────────────
const fmt = {
  dollar: (v: number | null | undefined) => {
    if (v == null) return '—';
    const abs = Math.abs(v);
    if (abs >= 1_000_000) return `${v < 0 ? '-' : ''}$${(abs / 1_000_000).toFixed(1)}M`;
    if (abs >= 1_000) return `${v < 0 ? '-' : ''}$${(abs / 1_000).toFixed(0)}K`;
    return `$${v.toFixed(0)}`;
  },
  war: (v: number | null | undefined) => (v != null ? v.toFixed(1) : '—'),
  pct: (v: number | null | undefined) => (v != null ? `${(v * 100).toFixed(1)}%` : '—'),
  dec: (v: number | null | undefined, d = 3) => (v != null ? v.toFixed(d) : '—'),
  int: (v: number | null | undefined) => (v != null ? Math.round(v).toString() : '—'),
};

/** Clamp to 0-1 range */
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

// ─── Subcomponents ──────────────────────────────────────────

/** Headshot with fallback silhouette — fitted, not zoomed */
const PlayerHeadshot: React.FC<{ mlbId: number | null; name: string; teamColor: string; size?: string }> = ({
  mlbId,
  name,
  teamColor,
  size = 'w-40 h-40 md:w-48 md:h-48',
}) => {
  const [imgError, setImgError] = useState(false);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!mlbId) return;
    let cancelled = false;
    const fetchImage = async () => {
      try {
        const response = await fetch(`${API_BASE}/headshots/${mlbId}.png`, {
          headers: { 'ngrok-skip-browser-warning': 'true', 'User-Agent': 'LongballAnalytics/1.0' },
        });
        if (!response.ok) { setImgError(true); return; }
        const blob = await response.blob();
        if (!cancelled) setBlobUrl(URL.createObjectURL(blob));
      } catch { if (!cancelled) setImgError(true); }
    };
    fetchImage();
    return () => { cancelled = true; };
  }, [mlbId]);

  // Revoke blob on unmount
  useEffect(() => () => { if (blobUrl) URL.revokeObjectURL(blobUrl); }, [blobUrl]);

  return (
    <div
      className={`relative ${size} rounded-2xl overflow-hidden shrink-0 ring-1 ring-white/10`}
      style={{ background: `linear-gradient(135deg, ${teamColor}18, ${teamColor}08)` }}
    >
      {blobUrl && !imgError ? (
        <img
          src={blobUrl}
          alt={name}
          className="w-full h-full object-contain object-bottom"
          onError={() => setImgError(true)}
        />
      ) : (
        <div className="w-full h-full flex items-center justify-center">
          <svg className="w-20 h-20 text-surface-600" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z" />
          </svg>
        </div>
      )}
    </div>
  );
};



/** ──────────────────────────────────────────────────────────
 *  Surplus Bar — horizontal value bar (-$50M → +$50M range)
 *  ────────────────────────────────────────────────────────── */
const SurplusBar: React.FC<{ value: number; teamColor: string }> = ({ value, teamColor }) => {
  const maxAbs = 200_000_000; // $200M range
  const pct = clamp01((value / maxAbs + 1) / 2); // 0 = -max, 0.5 = 0, 1 = +max
  const isPositive = value >= 0;

  return (
    <div className="w-full">
      <div className="flex justify-between items-baseline mb-1.5">
        <span className="text-xs text-surface-400 font-medium">Surplus Value</span>
        <span
          className="text-lg font-bold"
          style={{ color: isPositive ? teamColor : '#f87171' }}
        >
          {fmt.dollar(value)}
        </span>
      </div>
      <div className="relative h-2.5 bg-surface-700/60 rounded-full overflow-hidden">
        {/* Center line marker */}
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-surface-500/60 z-10" />
        {/* Fill bar */}
        <div
          className="absolute top-0 bottom-0 rounded-full transition-all duration-700 ease-out"
          style={{
            left: isPositive ? '50%' : `${pct * 100}%`,
            width: `${Math.abs(pct - 0.5) * 100}%`,
            background: isPositive
              ? `linear-gradient(90deg, ${teamColor}90, ${teamColor})`
              : 'linear-gradient(90deg, #f87171, #f8717190)',
          }}
        />
      </div>
      <div className="flex justify-between mt-1">
        <span className="text-[10px] text-surface-600">-$200M</span>
        <span className="text-[10px] text-surface-600">$0</span>
        <span className="text-[10px] text-surface-600">+$200M</span>
      </div>
    </div>
  );
};

/** ──────────────────────────────────────────────────────────
 *  Contract Timeline — visual year-by-year control bar
 *  ────────────────────────────────────────────────────────── */
const ContractTimeline: React.FC<{
  currentYear: number;
  faEarliest?: number | null;
  faProbable?: number | null;
  faLatest?: number | null;
  yearsControl?: number | null;
  teamColor: string;
  projections: Array<{ year: number; status: string }>;
}> = ({ currentYear, faEarliest, faProbable, faLatest, yearsControl, teamColor, projections }) => {
  const yrsCtrl = yearsControl ?? 0;
  const endYear = Math.max(
    currentYear + yrsCtrl,
    faLatest ?? currentYear,
    faProbable ?? currentYear,
    currentYear + 1
  );
  const years = Array.from({ length: endYear - currentYear + 1 }, (_, i) => currentYear + i);

  // Find the status for earliest FA year for legend
  const earliestFaProj = projections.find(p => p.year === faEarliest);
  const earliestFaStatus = earliestFaProj?.status || '';
  const isOptionYear = earliestFaStatus.toLowerCase().includes('option');
  const optionType = isOptionYear ? earliestFaStatus : 'Option Year';

  return (
    <div className="w-full">
      <div className="flex justify-between items-baseline mb-2">
        <span className="text-xs text-surface-400 font-medium">Contract Control</span>
        <span className="text-xs text-surface-500">
          {yrsCtrl > 0 ? `${yrsCtrl} yr${yrsCtrl > 1 ? 's' : ''} remaining` : 'Free Agent'}
        </span>
      </div>
      <div className="flex gap-1">
        {years.map((yr) => {
          const isControlled = yr < currentYear + yrsCtrl;
          const isFaProbable = yr === faProbable;
          const isFaEarliest = yr === faEarliest;
          const proj = projections.find(p => p.year === yr);
          const status = proj?.status || '';
          return (
            <div key={yr} className="flex-1 flex flex-col items-center gap-0.5">
              <div
                className="w-full h-6 rounded-md transition-all flex items-center justify-center"
                style={{
                  backgroundColor: isControlled
                    ? teamColor + (yr === currentYear ? 'FF' : '80')
                    : isFaProbable
                    ? '#f59e0b40'
                    : 'rgba(255,255,255,0.04)',
                  border: isFaEarliest ? '1px dashed #f59e0b' : isFaProbable ? '1px solid #f59e0b60' : '1px solid transparent',
                }}
              >
                {yr === currentYear && (
                  <div className="w-1.5 h-1.5 rounded-full bg-white" />
                )}
              </div>
              <span className="text-[9px] text-surface-500 tabular-nums">{yr.toString().slice(-2)}</span>
              {status && (
                <span className="text-[8px] text-surface-600 text-center leading-tight px-0.5" style={{ maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {status}
                </span>
              )}
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-3 mt-2">
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: teamColor }} />
          <span className="text-[10px] text-surface-500">Under Control</span>
        </div>
        {faEarliest && faEarliest !== faProbable && isOptionYear && (
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-sm bg-transparent" style={{ border: '1.5px dashed #f59e0b' }} />
            <span className="text-[10px] text-surface-500">{optionType}</span>
          </div>
        )}
        {faProbable && (
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-sm bg-amber-500/30 border border-amber-500/60" />
            <span className="text-[10px] text-surface-500">Probable FA</span>
          </div>
        )}
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-sm bg-white/[0.04] border border-white/[0.08]" />
          <span className="text-[10px] text-surface-500">Free Agent</span>
        </div>
      </div>
    </div>
  );
};



/** ──────────────────────────────────────────────────────────
 *  Collapsible Section wrapper
 *  ────────────────────────────────────────────────────────── */
const CollapsibleSection: React.FC<{
  title: string;
  teamColor: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}> = ({ title, teamColor, defaultOpen = false, children }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="rounded-xl overflow-hidden border border-white/[0.06] bg-surface-800/40">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-white/[0.02] transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-1 h-5 rounded-full" style={{ backgroundColor: teamColor }} />
          <h2 className="text-lg font-semibold text-white">{title}</h2>
        </div>
        <svg
          className={`w-5 h-5 text-surface-400 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {open && <div className="border-t border-white/[0.06]">{children}</div>}
    </section>
  );
};

// ─── Main Component ─────────────────────────────────────────

const PlayerDetails: React.FC = () => {
  const { playerId } = useParams<{ playerId: string }>();
  const [player, setPlayer] = useState<PlayerStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tradeHistory, setTradeHistory] = useState<TradeValuePoint[]>([]);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [playerInfo, setPlayerInfo] = useState<PlayerInfo | null>(null);
  const [pastTrades, setPastTrades] = useState<PastTradeDetail[]>([]);

  useEffect(() => {
    const fetchPlayer = async () => {
      if (!playerId) return;
      setLoading(true);
      try {
        const data = await getPlayerDetails(parseInt(playerId));
        setPlayer(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching player:', err);
        setError('Failed to load player details');
      } finally {
        setLoading(false);
      }
    };
    fetchPlayer();
  }, [playerId]);

  // Fetch trade value history once we have the player's mlb_id (or fallback to playerId)
  useEffect(() => {
    const id = player?.mlb_id ?? (playerId ? parseInt(playerId) : null);
    if (id == null) return;
    getTradeValueHistory(id).then(setTradeHistory).catch(() => setTradeHistory([]));
    getPlayerTransactions(id).then(setTransactions).catch(() => setTransactions([]));
    getPlayerInfo(id).then(setPlayerInfo).catch(() => setPlayerInfo(null));
    getPlayerPastTrades(id).then(r => setPastTrades(r.trades)).catch(() => setPastTrades([]));
  }, [player?.mlb_id, playerId]);

  // ── Derived state ──
  const isHistorical = player?.isHistorical === true;
  const histMeta = player?.historicalMeta;

  // For historical players, use last season as "current"; for projected players use CURRENT_YEAR
  const cur = isHistorical
    ? player?.projections[player.projections.length - 1]
    : (player?.projections.find((p) => p.year === CURRENT_YEAR) ?? player?.projections[0]);
  const team = cur?.team ?? player?.team ?? '';
  const colors = getTeamColors(team);
  const teamName = getTeamName(team);

  const MAX_PROJECTION_YEAR = CURRENT_YEAR + MAX_PROJECTION_YEARS - 1;

  const hasPitching = player?.projections.some((p) => p.pitching?.war_pit != null);
  const hasHitting = player?.projections.some((p) => p.hitting?.war_bat != null);

  const hasCurrentHitting = isHistorical
    ? hasHitting
    : player?.projections.some((p) => p.year === CURRENT_YEAR && p.hitting?.war_bat != null);
  const hasCurrentPitching = isHistorical
    ? hasPitching
    : player?.projections.some((p) => p.year === CURRENT_YEAR && p.pitching?.war_pit != null);

  const pitchingTableData = useMemo(() => {
    if (!player?.projections) return [];
    return player.projections
      .filter(
        (proj): proj is typeof proj & { pitching: NonNullable<typeof proj.pitching> } =>
          proj.pitching?.war_pit != null && (isHistorical || proj.year <= MAX_PROJECTION_YEAR)
      )
      .map((proj) => ({ year: proj.year, age: proj.age, team: proj.team, status: proj.status, value: proj.value, pitching: proj.pitching }));
  }, [player, MAX_PROJECTION_YEAR, isHistorical]);

  const hittingTableData = useMemo(() => {
    if (!player?.projections) return [];
    return player.projections
      .filter(
        (proj): proj is typeof proj & { hitting: NonNullable<typeof proj.hitting> } =>
          proj.hitting?.war_bat != null && (isHistorical || proj.year <= MAX_PROJECTION_YEAR)
      )
      .map((proj) => ({ year: proj.year, age: proj.age, team: proj.team, status: proj.status, value: proj.value, hitting: proj.hitting }));
  }, [player, MAX_PROJECTION_YEAR, isHistorical]);

  // ── Loading / Error ──
  if (loading) {
    return (
      <div className="min-h-screen flex justify-center items-center">
        <div
          className="animate-spin rounded-full h-10 w-10 border-4 border-t-transparent"
          style={{ borderColor: colors.primary, borderTopColor: 'transparent' }}
        />
      </div>
    );
  }

  if (error || !player) {
    return (
      <div className="min-h-screen flex justify-center items-center">
        <div className="rounded-lg px-6 py-4 border border-red-500/20 bg-red-500/10">
          <p className="text-red-400">{error || 'Player not found'}</p>
        </div>
      </div>
    );
  }

  // ── Shortcuts ──
  const faEarliest = cur?.earliest_fa_year;
  const faProbable = cur?.probable_fa_year;
  const faLatest = cur?.fa_year;
  const h = cur?.hitting;
  const pit = cur?.pitching;
  const v = cur?.value;

  const projWar = isHistorical ? (histMeta?.career_war ?? 0) : (v?.contract_war ?? 0);

  return (
    <div className="min-h-screen bg-surface-900">

      {/* ════════════════════════════════════════════════════
       *  HERO — Big cinematic header with headshot + WAR ring
       *  ════════════════════════════════════════════════════ */}
      <div className="relative overflow-hidden">
        {/* Layered team-color background */}
        <div className="absolute inset-0" style={{ background: colors.gradient, opacity: 0.18 }} />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(255,255,255,0.06),transparent)]" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-surface-900/50 to-surface-900" />

        <div className="relative max-w-6xl mx-auto px-4 pt-10 pb-6 md:pt-14 md:pb-8">
          {/* Top: team name breadcrumb */}
          <div className="flex items-center gap-2 mb-4">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: colors.primary }} />
            <span className="text-xs font-bold uppercase tracking-[0.2em]" style={{ color: colors.accent }}>
              {isHistorical ? (histMeta?.teams?.join(' / ') ?? teamName) : teamName}
            </span>
          </div>

          <div className="flex flex-col md:flex-row items-start gap-6 md:gap-10">
            {/* Left: Headshot */}
            <PlayerHeadshot
              mlbId={player.mlb_id}
              name={player.name}
              teamColor={colors.primary}
            />

            {/* Center: Identity + badges */}
            <div className="flex-1 min-w-0 pt-1">
              <h1 className="text-4xl md:text-5xl font-extrabold text-white tracking-tight mb-3">
                {player.name}
              </h1>

              <div className="flex flex-wrap gap-2 mb-5">
                <span
                  className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide border"
                  style={{
                    backgroundColor: colors.primary + '15',
                    borderColor: colors.primary + '40',
                    color: colors.accent,
                  }}
                >
                  {player.position}
                </span>
                {isHistorical && histMeta && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-white/[0.06] text-surface-300 border border-white/[0.08]">
                    {histMeta.first_year}&ndash;{histMeta.last_year}
                  </span>
                )}
                {isHistorical && histMeta?.death_year && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-white/[0.06] text-surface-400 border border-white/[0.08]">
                    {histMeta.birth_year}&ndash;{histMeta.death_year}
                  </span>
                )}
                {!isHistorical && cur?.age && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-white/[0.06] text-surface-300 border border-white/[0.08]">
                    Age {cur.age}
                  </span>
                )}
                {!isHistorical && cur?.status && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-white/[0.06] text-surface-300 border border-white/[0.08]">
                    {cur.status}
                  </span>
                )}
                {isHistorical && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                    Historical
                  </span>
                )}
              </div>

              {/* Surplus value bar — only for projected players */}
              {!isHistorical && v && <SurplusBar value={v.trade_value ?? 0} teamColor={colors.accent} />}

              {/* Career WAR summary for historical players */}
              {isHistorical && histMeta && (
                <div className="flex gap-6 mt-2">
                  <div>
                    <span className="text-[10px] uppercase tracking-widest text-surface-500 block mb-0.5">Career WAR</span>
                    <span className="text-3xl font-bold" style={{ color: colors.accent }}>{histMeta.career_war.toFixed(1)}</span>
                  </div>
                  {histMeta.career_bat_war > 0 && (
                    <div>
                      <span className="text-[10px] uppercase tracking-widest text-surface-500 block mb-0.5">Batting WAR</span>
                      <span className="text-xl font-bold text-white">{histMeta.career_bat_war.toFixed(1)}</span>
                    </div>
                  )}
                  {histMeta.career_pit_war > 0 && (
                    <div>
                      <span className="text-[10px] uppercase tracking-widest text-surface-500 block mb-0.5">Pitching WAR</span>
                      <span className="text-xl font-bold text-white">{histMeta.career_pit_war.toFixed(1)}</span>
                    </div>
                  )}
                  {histMeta.career_salary != null && (
                    <div>
                      <span className="text-[10px] uppercase tracking-widest text-surface-500 block mb-0.5">Career Earnings</span>
                      <span className="text-xl font-bold text-red-400">{fmt.dollar(histMeta.career_salary)}</span>
                    </div>
                  )}
                  {histMeta.career_surplus != null && (
                    <div>
                      <span className="text-[10px] uppercase tracking-widest text-surface-500 block mb-0.5">Career Surplus</span>
                      <span className={`text-xl font-bold ${histMeta.career_surplus >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {fmt.dollar(histMeta.career_surplus)}
                      </span>
                      <span className="text-[9px] text-surface-600 block">inflation-adjusted</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ════════════════════════════════════════════════════
       *  PLAYER BIO — bio, awards, draft info
       *  ════════════════════════════════════════════════════ */}
      {playerInfo && (
        <div className="max-w-6xl mx-auto px-4 mb-8">
          <div className="rounded-xl border border-white/[0.06] bg-surface-800/40 p-6">
            <PlayerBioSection info={playerInfo} teamColor={colors.accent} />
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════════
       *  VALUE & CONTRACT SECTION — only for projected players
       *  ════════════════════════════════════════════════════ */}
      {!isHistorical && (
      <div className="max-w-6xl mx-auto px-4 mb-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Left: Contract timeline + financial summary */}
          <div className="rounded-xl p-6 border border-white/[0.06] bg-surface-800/40 space-y-6">
            <ContractTimeline
              currentYear={CURRENT_YEAR}
              faEarliest={faEarliest}
              faProbable={faProbable}
              faLatest={faLatest}
              yearsControl={v?.years_control}
              teamColor={colors.primary}
              projections={player.projections.map(p => ({ year: p.year, status: p.status }))}
            />

            <div className="grid grid-cols-2 gap-4 pt-2">
              <div>
                <span className="text-[10px] uppercase tracking-widest text-surface-500 block mb-0.5">Total Contract</span>
                <span className="text-xl font-bold text-red-400">{fmt.dollar(v?.total_contract)}</span>
              </div>
              <div>
                <span className="text-[10px] uppercase tracking-widest text-surface-500 block mb-0.5">Avg $/Year</span>
                <span className="text-xl font-bold text-red-400">{fmt.dollar(v?.avg_contract)}</span>
              </div>
            </div>
          </div>

          {/* Right: Production + Career summary */}
          <div className="rounded-xl p-6 border border-white/[0.06] bg-surface-800/40">
            <h3 className="text-xs font-semibold uppercase tracking-widest text-surface-400 mb-5">
              Production Overview
            </h3>
            <div className="grid grid-cols-2 gap-x-8 gap-y-5">
              <div>
                <span className="text-[10px] uppercase tracking-widest text-surface-500 block mb-0.5">Projected WAR</span>
                <span className="text-2xl font-bold" style={{ color: colors.accent }}>{fmt.war(projWar)}</span>
                <span className="text-[10px] text-surface-500 block mt-0.5">
                  over {v?.years_control ?? '?'} yr · {fmt.war(v?.avg_war)} WAR/yr
                </span>
              </div>
              <div>
                <span className="text-[10px] uppercase tracking-widest text-surface-500 block mb-0.5">Projected Value</span>
                <span className="text-2xl font-bold text-white">{fmt.dollar(v?.contract_base_value)}</span>
                <span className="text-[10px] text-surface-500 block mt-0.5">on-field production value</span>
              </div>
              <div>
                <span className="text-[10px] uppercase tracking-widest text-surface-500 block mb-0.5">Historical WAR</span>
                <span className="text-2xl font-bold text-white">{fmt.war(v?.historical_war)}</span>
                <span className="text-[10px] text-surface-500 block mt-0.5">career to date</span>
              </div>
              <div>
                <span className="text-[10px] uppercase tracking-widest text-surface-500 block mb-0.5">Total WAR</span>
                <span className="text-2xl font-bold text-white">{fmt.war(v?.total_war)}</span>
                <span className="text-[10px] text-surface-500 block mt-0.5">hist + projected</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      )}

      {/* ════════════════════════════════════════════════════
       *  TRADE VALUE TIMELINE — rolling value chart
       *  ════════════════════════════════════════════════════ */}
      {tradeHistory.length > 0 && (
        <div className="max-w-6xl mx-auto px-4 mb-8">
          <div className="rounded-xl border border-white/[0.06] bg-surface-800/40 p-6 relative">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-1 h-5 rounded-full" style={{ backgroundColor: colors.primary }} />
              <h2 className="text-lg font-semibold text-white">Trade Value History</h2>
            </div>
            <TradeValueChart
              data={tradeHistory}
              teamColor={colors.primary}
              teamAccent={colors.accent}
            />
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════════
       *  STAT TABLES — "Projections" for active, "Career Statistics" for historical
       *  ════════════════════════════════════════════════════ */}
      <div className="max-w-6xl mx-auto px-4 pb-16 space-y-4">
        {hasPitching && hasCurrentPitching && (
          <CollapsibleSection title={isHistorical ? "Pitching Statistics" : "Pitching Projections"} teamColor={colors.primary} defaultOpen>
            <CombinedPitchingTable data={pitchingTableData.sort((a, b) => b.year - a.year)} />
          </CollapsibleSection>
        )}

        {hasHitting && hasCurrentHitting && (
          <CollapsibleSection title={isHistorical ? "Hitting Statistics" : "Hitting Projections"} teamColor={colors.primary} defaultOpen>
            <CombinedHittingTable data={hittingTableData.sort((a, b) => b.year - a.year)} />
          </CollapsibleSection>
        )}

        {pastTrades.length > 0 && (
          <CollapsibleSection title="Past Trades" teamColor={colors.primary} defaultOpen>
            <div className="p-4">
              <PlayerTradeHistory
                trades={pastTrades}
                playerMlbId={player?.mlb_id ?? (playerId ? parseInt(playerId) : 0)}
                teamColor={colors.primary}
              />
            </div>
          </CollapsibleSection>
        )}

        {transactions.length > 0 && (
          <CollapsibleSection title="Transaction History" teamColor={colors.primary} defaultOpen={false}>
            <div className="p-4">
              <TransactionHistory transactions={transactions} teamColor={colors.primary} />
            </div>
          </CollapsibleSection>
        )}
      </div>
    </div>
  );
};

export default PlayerDetails;
