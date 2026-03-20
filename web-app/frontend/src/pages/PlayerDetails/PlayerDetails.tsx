import React, { useState, useEffect, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { getPlayerDetails, getTradeValueHistory, getPlayerTransactions, getPlayerInfo, getPlayerPastTrades, getPlayerMiLBStats, PlayerStats } from '../../services/api';
import type { TradeValuePoint, Transaction, PlayerInfo, PastTradeDetail, MiLBStatsResponse, ProspectDetailHistory } from '../../services/api';
import { CURRENT_YEAR, MAX_PROJECTION_YEARS } from '../../config';
import { CombinedHittingTable, CombinedPitchingTable } from '../../components/Tables';
import { getTeamColors, getTeamName } from '../../utils/teamColors';
import TradeValueChart from '../../components/TradeValueChart';
import TransactionHistory from '../../components/TransactionHistory';
import PlayerBioSection from '../../components/PlayerBioSection';
import PlayerTradeHistory from '../../components/PlayerTradeHistory';
import MiLBStatsTable from '../../components/MiLBStatsTable';

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

/** Headshot from MLB CDN — uses mlbam_id directly, no local files needed */
const MLB_HEADSHOT_URL = (mlbId: number) =>
  `https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/${mlbId}/headshot/67/current`;

const PlayerHeadshot: React.FC<{ mlbId: number | null; name: string; teamColor: string; size?: string }> = ({
  mlbId,
  name,
  teamColor,
  size = 'w-40 md:w-48',
}) => {
  const [imgError, setImgError] = useState(false);

  return (
    <div
      className={`relative ${size} aspect-[79/119] rounded-2xl overflow-hidden shrink-0`}
      style={{ background: `linear-gradient(135deg, ${teamColor}18, ${teamColor}08)` }}
    >
      {mlbId && !imgError ? (
        <img
          src={MLB_HEADSHOT_URL(mlbId)}
          alt={name}
          className="w-full h-full object-cover object-top"
          onError={() => setImgError(true)}
        />
      ) : (
        <div className="w-full h-full flex items-center justify-center">
          <svg className="w-20 h-20 text-gray-500" fill="currentColor" viewBox="0 0 24 24">
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
        <span className="text-xs text-gray-500 font-medium">Surplus Value</span>
        <span
          className="text-lg font-bold"
          style={{ color: isPositive ? teamColor : '#f87171' }}
        >
          {fmt.dollar(value)}
        </span>
      </div>
      <div className="relative h-2.5 bg-gray-100/60 rounded-full overflow-hidden">
        {/* Center line marker */}
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-gray-300 z-10" />
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
        <span className="text-[10px] text-gray-500">-$200M</span>
        <span className="text-[10px] text-gray-500">$0</span>
        <span className="text-[10px] text-gray-500">+$200M</span>
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
        <span className="text-xs text-gray-500 font-medium">Contract Control</span>
        <span className="text-xs text-gray-500">
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
              <span className="text-[9px] text-gray-400 tabular-nums">{yr.toString().slice(-2)}</span>
              {status && (
                <span className="text-[8px] text-gray-400 text-center leading-tight px-0.5" style={{ maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
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
          <span className="text-[10px] text-gray-500">Under Control</span>
        </div>
        {faEarliest && faEarliest !== faProbable && isOptionYear && (
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-sm bg-transparent" style={{ border: '1.5px dashed #f59e0b' }} />
            <span className="text-[10px] text-gray-500">{optionType}</span>
          </div>
        )}
        {faProbable && (
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-sm bg-amber-500/30 border border-amber-500/60" />
            <span className="text-[10px] text-gray-500">Probable FA</span>
          </div>
        )}
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-sm bg-gray-50 border border-gray-200" />
          <span className="text-[10px] text-gray-500">Free Agent</span>
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
    <section className="rounded-xl overflow-hidden border border-gray-200 bg-gray-50">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-white/[0.02] transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-1 h-5 rounded-full" style={{ backgroundColor: teamColor }} />
          <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
        </div>
        <svg
          className={`w-5 h-5 text-gray-500 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {open && <div className="border-t border-gray-200">{children}</div>}
    </section>
  );
};

// ─── Prospect helpers ───────────────────────────────────────

const fvColor = (fv: string) => {
  const n = parseInt(fv);
  if (n >= 70) return 'text-emerald-400';
  if (n >= 60) return 'text-blue-400';
  if (n >= 55) return 'text-sky-400';
  if (n >= 50) return 'text-amber-400';
  if (n >= 45) return 'text-orange-400';
  return 'text-gray-500';
};

const fvBg = (fv: string) => {
  const n = parseInt(fv);
  if (n >= 70) return 'bg-emerald-500/10 border-emerald-500/20';
  if (n >= 60) return 'bg-blue-500/10 border-blue-500/20';
  if (n >= 55) return 'bg-sky-500/10 border-sky-500/20';
  if (n >= 50) return 'bg-amber-500/10 border-amber-500/20';
  if (n >= 45) return 'bg-orange-500/10 border-orange-500/20';
  return 'bg-white border-gray-200';
};

const gradeColor = (grade: string | null | undefined) => {
  if (!grade) return 'text-gray-500';
  const n = parseInt(grade);
  if (isNaN(n)) return 'text-gray-500';
  if (n >= 70) return 'text-emerald-400';
  if (n >= 60) return 'text-blue-400';
  if (n >= 55) return 'text-sky-400';
  if (n >= 50) return 'text-gray-800';
  if (n >= 45) return 'text-amber-400';
  if (n >= 40) return 'text-orange-400';
  return 'text-red-400';
};

const GradeBar: React.FC<{ label: string; grade: string | null | undefined }> = ({ label, grade }) => {
  const value = grade ? parseInt(grade) : 0;
  const pct = Math.min(100, Math.max(0, ((value - 20) / 60) * 100));
  return (
    <div className="flex items-center gap-3">
      <span className="w-20 text-[11px] text-gray-500 text-right">{label}</span>
      <div className="flex-1 h-2 bg-white rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{
            width: `${pct}%`,
            backgroundColor: value >= 60 ? '#60a5fa' : value >= 50 ? '#fbbf24' : '#f97316',
          }}
        />
      </div>
      <span className={`w-8 text-[12px] font-mono font-semibold text-right ${gradeColor(grade)}`}>
        {grade ?? '-'}
      </span>
    </div>
  );
};

/** Prospect Profile section — collapsible, shows tool grades, FV, rankings, and year history */
const ProspectProfile: React.FC<{
  prospectData: NonNullable<PlayerStats['prospectData']>;
  teamColor: string;
}> = ({ prospectData, teamColor }) => {
  const { tools, history, fv, is_pitcher } = prospectData;
  const latest = history[0];
  // Compute peak Top 100 value across all prospect years
  const peakTop100Value = history.reduce<number | null>((best, h) => {
    if (h.value != null && h.top_100 != null) {
      return best == null ? h.value : Math.max(best, h.value);
    }
    return best;
  }, null);
  // Year range the player was a prospect
  const years = history.map(h => h.year).sort((a, b) => a - b);
  const yearLabel = years.length === 1 ? `${years[0]}` : `${years[0]}–${years[years.length - 1]}`;
  return (
    <div className="p-6 space-y-6">
      {/* FV + ranking badges */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className={`flex flex-col items-center px-4 py-2 rounded-lg border ${fvBg(fv)}`}>
          <span className="text-[10px] uppercase tracking-wider text-gray-400 mb-0.5">FV</span>
          <span className={`text-2xl font-bold ${fvColor(fv)}`}>{fv}</span>
        </div>
        {latest?.top_100 && (
          <div className="flex flex-col items-center px-4 py-2 rounded-lg border bg-amber-500/10 border-amber-500/20">
            <span className="text-[10px] uppercase tracking-wider text-gray-400 mb-0.5">Top 100</span>
            <span className="text-2xl font-bold text-amber-400">#{latest.top_100}</span>
          </div>
        )}
        {peakTop100Value != null && (
          <div className="flex flex-col items-center px-4 py-2 rounded-lg border bg-emerald-500/10 border-emerald-500/20">
            <span className="text-[10px] uppercase tracking-wider text-gray-400 mb-0.5">Top 100 Value</span>
            <span className="text-2xl font-bold text-emerald-400">${(peakTop100Value / 1_000_000).toFixed(1)}M</span>
          </div>
        )}
        {latest?.org_rank && (
          <div className="flex flex-col items-center px-4 py-2 rounded-lg border bg-white border-gray-200">
            <span className="text-[10px] uppercase tracking-wider text-gray-400 mb-0.5">Org Rank</span>
            <span className="text-2xl font-bold text-gray-800">#{latest.org_rank}</span>
          </div>
        )}
        <div className="flex flex-col items-center px-4 py-2 rounded-lg border bg-white border-gray-200">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 mb-0.5">Prospect Years</span>
          <span className="text-lg font-bold text-gray-600">{yearLabel}</span>
        </div>
      </div>

      {/* Tool grades */}
      <div>
        <h3 className="text-[11px] uppercase tracking-wider text-gray-400 mb-3">
          {is_pitcher ? 'Pitch Grades' : 'Tool Grades'}
        </h3>
        <div className="space-y-2.5 max-w-md">
          {is_pitcher ? (
            <>
              <GradeBar label="Fastball" grade={tools.fastball} />
              <GradeBar label="Slider" grade={tools.slider} />
              <GradeBar label="Curve" grade={tools.curve} />
              <GradeBar label="Changeup" grade={tools.changeup} />
              <GradeBar label="Command" grade={tools.command} />
            </>
          ) : (
            <>
              <GradeBar label="Hit" grade={tools.hit} />
              <GradeBar label="Game Power" grade={tools.game_power} />
              <GradeBar label="Raw Power" grade={tools.raw_power} />
              <GradeBar label="Speed" grade={tools.speed} />
            </>
          )}
        </div>
      </div>

      {/* Year-by-year ranking history */}
      {history.length > 1 && (
        <div>
          <h3 className="text-[11px] uppercase tracking-wider text-gray-400 mb-3">Ranking History</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="px-2 py-2 text-left text-[10px] text-gray-400 uppercase tracking-wider">Year</th>
                  <th className="px-2 py-2 text-left text-[10px] text-gray-400 uppercase tracking-wider">Org</th>
                  <th className="px-2 py-2 text-left text-[10px] text-gray-400 uppercase tracking-wider">FV</th>
                  <th className="px-2 py-2 text-center text-[10px] text-gray-400 uppercase tracking-wider">T100</th>
                  <th className="px-2 py-2 text-center text-[10px] text-gray-400 uppercase tracking-wider">Org#</th>
                  <th className="px-2 py-2 text-right text-[10px] text-gray-400 uppercase tracking-wider">Value</th>
                  {is_pitcher ? (
                    <>
                      <th className="px-2 py-2 text-center text-[10px] text-gray-400 uppercase tracking-wider">FB</th>
                      <th className="px-2 py-2 text-center text-[10px] text-gray-400 uppercase tracking-wider">SL</th>
                      <th className="px-2 py-2 text-center text-[10px] text-gray-400 uppercase tracking-wider">CB</th>
                      <th className="px-2 py-2 text-center text-[10px] text-gray-400 uppercase tracking-wider">CH</th>
                      <th className="px-2 py-2 text-center text-[10px] text-gray-400 uppercase tracking-wider">CMD</th>
                    </>
                  ) : (
                    <>
                      <th className="px-2 py-2 text-center text-[10px] text-gray-400 uppercase tracking-wider">Hit</th>
                      <th className="px-2 py-2 text-center text-[10px] text-gray-400 uppercase tracking-wider">Game</th>
                      <th className="px-2 py-2 text-center text-[10px] text-gray-400 uppercase tracking-wider">Raw</th>
                      <th className="px-2 py-2 text-center text-[10px] text-gray-400 uppercase tracking-wider">Spd</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {history.map((h: ProspectDetailHistory, i: number) => (
                  <tr
                    key={h.year}
                    className={`text-[11px] border-b border-gray-100 ${i % 2 === 0 ? 'bg-transparent' : 'bg-gray-50/50'}`}
                  >
                    <td className="px-2 py-1.5 text-gray-800 font-medium">{h.year}</td>
                    <td className="px-2 py-1.5 text-gray-500">{h.org}</td>
                    <td className={`px-2 py-1.5 font-semibold ${fvColor(h.fv)}`}>{h.fv}</td>
                    <td className="px-2 py-1.5 text-center font-mono text-amber-400">{h.top_100 ? '#' + h.top_100 : '-'}</td>
                    <td className="px-2 py-1.5 text-center font-mono text-gray-500">{h.org_rank ? '#' + h.org_rank : '-'}</td>
                    <td className="px-2 py-1.5 text-right font-mono text-emerald-400">{h.value != null ? `$${(h.value / 1_000_000).toFixed(1)}M` : '-'}</td>
                    {is_pitcher ? (
                      <>
                        <td className={`px-2 py-1.5 text-center font-mono ${gradeColor(h.fastball)}`}>{h.fastball ?? '-'}</td>
                        <td className={`px-2 py-1.5 text-center font-mono ${gradeColor(h.slider)}`}>{h.slider ?? '-'}</td>
                        <td className={`px-2 py-1.5 text-center font-mono ${gradeColor(h.curve)}`}>{h.curve ?? '-'}</td>
                        <td className={`px-2 py-1.5 text-center font-mono ${gradeColor(h.changeup)}`}>{h.changeup ?? '-'}</td>
                        <td className={`px-2 py-1.5 text-center font-mono ${gradeColor(h.command)}`}>{h.command ?? '-'}</td>
                      </>
                    ) : (
                      <>
                        <td className={`px-2 py-1.5 text-center font-mono ${gradeColor(h.hit)}`}>{h.hit ?? '-'}</td>
                        <td className={`px-2 py-1.5 text-center font-mono ${gradeColor(h.game_power)}`}>{h.game_power ?? '-'}</td>
                        <td className={`px-2 py-1.5 text-center font-mono ${gradeColor(h.raw_power)}`}>{h.raw_power ?? '-'}</td>
                        <td className={`px-2 py-1.5 text-center font-mono ${gradeColor(h.speed)}`}>{h.speed ?? '-'}</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
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
  const [milbStats, setMilbStats] = useState<MiLBStatsResponse | null>(null);

  useEffect(() => {
    const fetchPlayer = async () => {
      if (!playerId) return;
      setLoading(true);
      try {
        const data = await getPlayerDetails(parseInt(playerId));
        setPlayer(data);
        setError(null);
      } catch (err) {
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
    getPlayerMiLBStats(id).then(setMilbStats).catch(() => setMilbStats(null));
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

  // Determine if this is primarily a pitcher (has pitching but position is a pitcher variant)
  const isPrimarilyPitcher = hasPitching && (
    player?.position === 'P' ||
    player?.position === 'SP' ||
    player?.position === 'RP' ||
    player?.position === 'CL' ||
    (isHistorical && histMeta?.career_pit_war != null && histMeta.career_pit_war > (histMeta?.career_bat_war ?? 0))
  );

  // For pitchers with hitting data: check if they have meaningful PA (g_bat > 0 in any year)
  const hittingHasGames = player?.projections.some(
    (p) => p.hitting?.g_bat != null && p.hitting.g_bat > 0
  );
  // Hide hitting entirely for pitchers with no games batted; collapse it if they have some
  const showHitting = hasHitting && (!isPrimarilyPitcher || hittingHasGames);
  const hittingDefaultOpen = !isPrimarilyPitcher;

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
          proj.pitching?.war_pit != null
          && (proj.pitching.g_pit > 0 || proj.pitching.ip > 0)
          && (isHistorical || proj.year <= MAX_PROJECTION_YEAR)
      )
      .map((proj) => ({ year: proj.year, age: proj.age, team: proj.team, status: proj.status, value: proj.value, pitching: proj.pitching }));
  }, [player, MAX_PROJECTION_YEAR, isHistorical]);

  const hittingTableData = useMemo(() => {
    if (!player?.projections) return [];
    return player.projections
      .filter(
        (proj): proj is typeof proj & { hitting: NonNullable<typeof proj.hitting> } =>
          proj.hitting?.war_bat != null
          && proj.hitting.g_bat > 0
          && (isHistorical || proj.year <= MAX_PROJECTION_YEAR)
      )
      .map((proj) => ({ year: proj.year, age: proj.age, team: proj.team, status: proj.status, value: proj.value, hitting: proj.hitting }));
  }, [player, MAX_PROJECTION_YEAR, isHistorical]);

  // Split historical vs projected data for separate tables
  const historicalPitching = useMemo(() => pitchingTableData.filter(d => d.year < CURRENT_YEAR), [pitchingTableData]);
  const projectedPitching = useMemo(() => pitchingTableData.filter(d => d.year >= CURRENT_YEAR), [pitchingTableData]);
  const historicalHitting = useMemo(() => hittingTableData.filter(d => d.year < CURRENT_YEAR), [hittingTableData]);
  const projectedHitting = useMemo(() => hittingTableData.filter(d => d.year >= CURRENT_YEAR), [hittingTableData]);

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
    <div className="min-h-screen bg-\[#F5F3EE\]">

      {/* ════════════════════════════════════════════════════
       *  HERO — Big cinematic header with headshot + WAR ring
       *  ════════════════════════════════════════════════════ */}
      <div className="relative overflow-hidden">
        {/* Layered team-color background */}
        <div className="absolute inset-0" style={{ background: colors.gradient, opacity: 0.18 }} />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(255,255,255,0.06),transparent)]" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-gray-50/80 to-\[#F5F3EE\]" />

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
              <h1 className="text-4xl md:text-5xl font-extrabold text-gray-900 tracking-tight mb-3">
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
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-gray-50 text-gray-600 border border-gray-200">
                    {histMeta.first_year}&ndash;{histMeta.last_year}
                  </span>
                )}
                {isHistorical && histMeta?.death_year && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-gray-50 text-gray-500 border border-gray-200">
                    {histMeta.birth_year}&ndash;{histMeta.death_year}
                  </span>
                )}
                {!isHistorical && cur?.age && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-gray-50 text-gray-600 border border-gray-200">
                    Age {cur.age}
                  </span>
                )}
                {!isHistorical && !cur?.age && player.prospectData?.age && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-gray-50 text-gray-600 border border-gray-200">
                    Age {Math.floor(player.prospectData.age)}
                  </span>
                )}
                {!isHistorical && cur?.status && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-gray-50 text-gray-600 border border-gray-200">
                    {cur.status}
                  </span>
                )}
                {player.isProspectOnly && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20">
                    Prospect
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
                    <span className="text-[10px] uppercase tracking-widest text-gray-400 block mb-0.5">Career WAR</span>
                    <span className="text-3xl font-bold" style={{ color: colors.accent }}>{histMeta.career_war.toFixed(1)}</span>
                  </div>
                  {histMeta.career_bat_war > 0 && (
                    <div>
                      <span className="text-[10px] uppercase tracking-widest text-gray-400 block mb-0.5">Batting WAR</span>
                      <span className="text-xl font-bold text-gray-900">{histMeta.career_bat_war.toFixed(1)}</span>
                    </div>
                  )}
                  {histMeta.career_pit_war > 0 && (
                    <div>
                      <span className="text-[10px] uppercase tracking-widest text-gray-400 block mb-0.5">Pitching WAR</span>
                      <span className="text-xl font-bold text-gray-900">{histMeta.career_pit_war.toFixed(1)}</span>
                    </div>
                  )}
                  {histMeta.career_salary != null && (
                    <div>
                      <span className="text-[10px] uppercase tracking-widest text-gray-400 block mb-0.5">Career Earnings</span>
                      <span className="text-xl font-bold text-red-400">{fmt.dollar(histMeta.career_salary)}</span>
                    </div>
                  )}
                  {histMeta.career_surplus != null && (
                    <div>
                      <span className="text-[10px] uppercase tracking-widest text-gray-400 block mb-0.5">Career Surplus</span>
                      <span className={`text-xl font-bold ${histMeta.career_surplus >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {fmt.dollar(histMeta.career_surplus)}
                      </span>
                      <span className="text-[9px] text-gray-400 block">inflation-adjusted</span>
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
          <div className="rounded-xl border border-gray-200 bg-gray-50 p-6">
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
          <div className="rounded-xl p-6 border border-gray-200 bg-gray-50 space-y-6">
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
                <span className="text-[10px] uppercase tracking-widest text-gray-400 block mb-0.5">Total Contract</span>
                <span className="text-xl font-bold text-red-400">{fmt.dollar(v?.total_contract)}</span>
              </div>
              <div>
                <span className="text-[10px] uppercase tracking-widest text-gray-400 block mb-0.5">Avg $/Year</span>
                <span className="text-xl font-bold text-red-400">{fmt.dollar(v?.avg_contract)}</span>
              </div>
            </div>
          </div>

          {/* Right: Production + Career summary */}
          <div className="rounded-xl p-6 border border-gray-200 bg-gray-50">
            <h3 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-5">
              Production Overview
            </h3>
            <div className="grid grid-cols-2 gap-x-8 gap-y-5">
              <div>
                <span className="text-[10px] uppercase tracking-widest text-gray-400 block mb-0.5">Projected WAR</span>
                <span className="text-2xl font-bold" style={{ color: colors.accent }}>{fmt.war(projWar)}</span>
                <span className="text-[10px] text-gray-400 block mt-0.5">
                  over {v?.years_control ?? '?'} yr · {fmt.war(v?.avg_war)} WAR/yr
                </span>
              </div>
              <div>
                <span className="text-[10px] uppercase tracking-widest text-gray-400 block mb-0.5">Projected Value</span>
                <span className="text-2xl font-bold text-gray-900">{fmt.dollar(v?.contract_base_value)}</span>
                <span className="text-[10px] text-gray-400 block mt-0.5">on-field production value</span>
              </div>
              <div>
                <span className="text-[10px] uppercase tracking-widest text-gray-400 block mb-0.5">Historical WAR</span>
                <span className="text-2xl font-bold text-gray-900">{fmt.war(v?.historical_war)}</span>
                <span className="text-[10px] text-gray-400 block mt-0.5">career to date</span>
              </div>
              <div>
                <span className="text-[10px] uppercase tracking-widest text-gray-400 block mb-0.5">Total WAR</span>
                <span className="text-2xl font-bold text-gray-900">{fmt.war(v?.total_war)}</span>
                <span className="text-[10px] text-gray-400 block mt-0.5">hist + projected</span>
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
          <div className="rounded-xl border border-gray-200 bg-gray-50 p-6 relative">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-1 h-5 rounded-full" style={{ backgroundColor: colors.primary }} />
              <h2 className="text-lg font-semibold text-gray-900">Trade Value History</h2>
            </div>
            <TradeValueChart
              data={tradeHistory}
              teamColor={colors.primary}
              teamAccent={colors.accent}
              transactions={transactions}
            />
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════════
       *  STAT TABLES — "Projections" for active, "Career Statistics" for historical
       *  ════════════════════════════════════════════════════ */}
      <div className="max-w-6xl mx-auto px-4 pb-16 space-y-4">
        {hasPitching && historicalPitching.length > 0 && (
          <CollapsibleSection title="Pitching Statistics" teamColor={colors.primary} defaultOpen>
            <CombinedPitchingTable data={historicalPitching.sort((a, b) => b.year - a.year)} showCareerTotals />
          </CollapsibleSection>
        )}

        {hasPitching && hasCurrentPitching && projectedPitching.length > 0 && !isHistorical && (
          <CollapsibleSection title="Pitching Projections" teamColor={colors.primary} defaultOpen>
            <CombinedPitchingTable data={projectedPitching.sort((a, b) => a.year - b.year)} hideXStats />
          </CollapsibleSection>
        )}

        {showHitting && historicalHitting.length > 0 && (
          <CollapsibleSection title="Hitting Statistics" teamColor={colors.primary} defaultOpen={hittingDefaultOpen}>
            <CombinedHittingTable data={historicalHitting.sort((a, b) => b.year - a.year)} showCareerTotals />
          </CollapsibleSection>
        )}

        {showHitting && hasCurrentHitting && projectedHitting.length > 0 && !isHistorical && (
          <CollapsibleSection title="Hitting Projections" teamColor={colors.primary} defaultOpen={hittingDefaultOpen}>
            <CombinedHittingTable data={projectedHitting.sort((a, b) => a.year - b.year)} hideXStats />
          </CollapsibleSection>
        )}

        {milbStats && (milbStats.hitting.length > 0 || milbStats.pitching.length > 0) && (
          <CollapsibleSection title="Minor League Statistics" teamColor={colors.primary} defaultOpen={false}>
            <div className="p-4">
              <MiLBStatsTable stats={milbStats} expandLatest />
            </div>
          </CollapsibleSection>
        )}

        {player.prospectData && (
          <CollapsibleSection
            title="Prospect Profile"
            teamColor={colors.primary}
            defaultOpen={!!player.isProspectOnly}
          >
            <ProspectProfile prospectData={player.prospectData} teamColor={colors.primary} />
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
