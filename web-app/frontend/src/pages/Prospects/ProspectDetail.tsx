import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  getProspectDetail,
  ProspectDetail as ProspectDetailType,
  getProspectMiLBStats,
  MiLBStatsResponse,
} from '../../services/api';
import { getTeamColors } from '../../utils/teamColors';
import MiLBStatsTable from '../../components/MiLBStatsTable';

// ── Helpers ──────────────────────────────────────────────────────────────────

const fvColor = (fv: string) => {
  const n = parseInt(fv);
  if (n >= 70) return 'text-emerald-400';
  if (n >= 60) return 'text-blue-400';
  if (n >= 55) return 'text-sky-400';
  if (n >= 50) return 'text-amber-400';
  if (n >= 45) return 'text-orange-400';
  return 'text-surface-400';
};

const fvBg = (fv: string) => {
  const n = parseInt(fv);
  if (n >= 70) return 'bg-emerald-500/10 border-emerald-500/20';
  if (n >= 60) return 'bg-blue-500/10 border-blue-500/20';
  if (n >= 55) return 'bg-sky-500/10 border-sky-500/20';
  if (n >= 50) return 'bg-amber-500/10 border-amber-500/20';
  if (n >= 45) return 'bg-orange-500/10 border-orange-500/20';
  return 'bg-surface-800 border-white/[0.06]';
};

const gradeColor = (grade: string | null | undefined) => {
  if (!grade) return 'text-surface-600';
  const n = parseInt(grade);
  if (isNaN(n)) return 'text-surface-400';
  if (n >= 70) return 'text-emerald-400';
  if (n >= 60) return 'text-blue-400';
  if (n >= 55) return 'text-sky-400';
  if (n >= 50) return 'text-surface-200';
  if (n >= 45) return 'text-amber-400';
  if (n >= 40) return 'text-orange-400';
  return 'text-red-400';
};

const formatValue = (v: number | null | undefined) => {
  if (v == null) return '-';
  return `$${(v / 1_000_000).toFixed(1)}M`;
};

// Prospect team codes to standard team map
const prospectTeamToStandard: Record<string, string> = {
  'SFG': 'SF', 'SDP': 'SD', 'KCR': 'KC', 'TBR': 'TB', 'OAK': 'ATH',
};

const getProspectTeamColors = (org: string) => {
  const mapped = prospectTeamToStandard[org] || org;
  return getTeamColors(mapped);
};

// ── Tool grade bar ──────────────────────────────────────────────────────────

function GradeBar({ label, grade }: { label: string; grade: string | null | undefined }) {
  const value = grade ? parseInt(grade) : 0;
  const pct = Math.min(100, Math.max(0, ((value - 20) / 60) * 100));

  return (
    <div className="flex items-center gap-3">
      <span className="w-20 text-[11px] text-surface-400 text-right">{label}</span>
      <div className="flex-1 h-2 bg-surface-800 rounded-full overflow-hidden">
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
}

// ── Main component ──────────────────────────────────────────────────────────

export default function ProspectDetailPage() {
  const { prospectId } = useParams<{ prospectId: string }>();
  const [prospect, setProspect] = useState<ProspectDetailType | null>(null);
  const [milbStats, setMilbStats] = useState<MiLBStatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!prospectId) return;
    const id = Number(prospectId);
    setLoading(true);
    Promise.all([
      getProspectDetail(id),
      getProspectMiLBStats(id),
    ])
      .then(([p, stats]) => {
        setProspect(p);
        setMilbStats(stats);
      })
      .catch((e) => setError(e.message || 'Prospect not found'))
      .finally(() => setLoading(false));
  }, [prospectId]);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <div className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
      </div>
    );
  }

  if (error || !prospect) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-12 text-center">
        <h2 className="text-lg font-bold text-surface-200 mb-2">Prospect Not Found</h2>
        <p className="text-surface-500 text-sm mb-4">{error}</p>
        <Link to="/prospects" className="text-blue-400 hover:text-blue-300 text-sm">
          Back to Prospects
        </Link>
      </div>
    );
  }

  const colors = getProspectTeamColors(prospect.org);
  const tools = prospect.tools;
  const latest = prospect.history[0];

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Back link */}
      <Link
        to="/prospects"
        className="inline-flex items-center gap-1 text-[12px] text-surface-500 hover:text-surface-300 mb-6 transition-colors"
      >
        <span>&larr;</span> Prospects
      </Link>

      {/* Header */}
      <div className="flex items-start gap-5 mb-8">
        {/* Headshot — use mlb_info.headshot_url which is built from mlbam_id */}
        {prospect.mlb_info?.headshot_url ? (
          <img
            src={prospect.mlb_info.headshot_url}
            alt={prospect.name}
            className="w-28 h-28 rounded-lg object-cover bg-surface-800"
            onError={(e) => {
              // Hide on error, show initial instead
              (e.target as HTMLImageElement).style.display = 'none';
              (e.target as HTMLImageElement).nextElementSibling?.classList.remove('hidden');
            }}
          />
        ) : null}
        <div
          className={`w-28 h-28 rounded-lg flex items-center justify-center text-3xl font-bold ${prospect.mlb_info?.headshot_url ? 'hidden' : ''}`}
          style={{ backgroundColor: colors.primary + '15', color: colors.primary }}
        >
          {prospect.name.charAt(0)}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-1">
            <h1 className="text-2xl font-bold text-surface-100 truncate">{prospect.name}</h1>
            {prospect.has_mlb && prospect.mlb_info?.mlb_id && (
              <Link
                to={`/players/${prospect.mlb_info.mlb_id}`}
                className="text-[10px] px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/15 hover:bg-blue-500/20 transition-colors"
              >
                View MLB Stats
              </Link>
            )}
          </div>
          <div className="flex items-center gap-2 text-[13px] text-surface-400">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: colors.primary }} />
            <span>{prospect.org}</span>
            <span className="text-surface-600">·</span>
            <span>{prospect.position}</span>
            {prospect.age && (
              <>
                <span className="text-surface-600">·</span>
                <span>Age {Math.floor(prospect.age)}</span>
              </>
            )}
          </div>
        </div>

        {/* FV + ranking badges */}
        <div className="flex items-start gap-3">
          <div className={`flex flex-col items-center px-4 py-2 rounded-lg border ${fvBg(prospect.fv)}`}>
            <span className="text-[10px] uppercase tracking-wider text-surface-500 mb-0.5">FV</span>
            <span className={`text-2xl font-bold ${fvColor(prospect.fv)}`}>{prospect.fv}</span>
          </div>
          {latest?.top_100 && (
            <div className="flex flex-col items-center px-4 py-2 rounded-lg border bg-amber-500/10 border-amber-500/20">
              <span className="text-[10px] uppercase tracking-wider text-surface-500 mb-0.5">Top 100</span>
              <span className="text-2xl font-bold text-amber-400">#{latest.top_100}</span>
            </div>
          )}
          {latest?.org_rank && (
            <div className="flex flex-col items-center px-4 py-2 rounded-lg border bg-surface-800 border-white/[0.06]">
              <span className="text-[10px] uppercase tracking-wider text-surface-500 mb-0.5">Org Rank</span>
              <span className="text-2xl font-bold text-surface-200">#{latest.org_rank}</span>
            </div>
          )}
        </div>
      </div>

      {/* Two-column layout: Tools + Value */}
      <div className="grid grid-cols-2 gap-8 mb-8">
        {/* Tool grades */}
        <div>
          <h2 className="text-[11px] uppercase tracking-wider text-surface-500 mb-3">
            {prospect.is_pitcher ? 'Pitch Grades' : 'Tool Grades'}
          </h2>
          <div className="space-y-2.5">
            {prospect.is_pitcher ? (
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

        {/* Value info */}
        <div>
          <h2 className="text-[11px] uppercase tracking-wider text-surface-500 mb-3">Valuation</h2>
          {latest && (
            <div className="space-y-3">
              <div className="flex justify-between items-center text-[13px]">
                <span className="text-surface-400">Model Value</span>
                <span className="text-surface-100 font-semibold">{formatValue(latest.value)}</span>
              </div>
              <div className="flex justify-between items-center text-[13px]">
                <span className="text-surface-400">FV Grade</span>
                <span className={`font-semibold ${fvColor(prospect.fv)}`}>{prospect.fv}</span>
              </div>
              {latest?.top_100 && (
                <div className="flex justify-between items-center text-[13px]">
                  <span className="text-surface-400">Top 100 Rank</span>
                  <span className="text-amber-400 font-semibold">#{latest.top_100}</span>
                </div>
              )}
              {latest?.org_rank && (
                <div className="flex justify-between items-center text-[13px]">
                  <span className="text-surface-400">Org Rank</span>
                  <span className="text-surface-100 font-semibold">#{latest.org_rank}</span>
                </div>
              )}
              <div className="flex justify-between items-center text-[13px]">
                <span className="text-surface-400">Organization</span>
                <span className="text-surface-100">{prospect.org}</span>
              </div>
              <div className="flex justify-between items-center text-[13px]">
                <span className="text-surface-400">Position</span>
                <span className="text-surface-100">{prospect.position}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Year-by-year ranking history */}
      {prospect.history.length > 1 && (
        <div className="mb-8">
          <h2 className="text-[11px] uppercase tracking-wider text-surface-500 mb-3">Ranking History</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-white/[0.06]">
                  <th className="px-2 py-2 text-left text-[10px] text-surface-500 uppercase tracking-wider">Year</th>
                  <th className="px-2 py-2 text-left text-[10px] text-surface-500 uppercase tracking-wider">Age</th>
                  <th className="px-2 py-2 text-left text-[10px] text-surface-500 uppercase tracking-wider">Org</th>
                  <th className="px-2 py-2 text-left text-[10px] text-surface-500 uppercase tracking-wider">Pos</th>
                  <th className="px-2 py-2 text-left text-[10px] text-surface-500 uppercase tracking-wider">FV</th>
                  <th className="px-2 py-2 text-center text-[10px] text-surface-500 uppercase tracking-wider">T100</th>
                  <th className="px-2 py-2 text-center text-[10px] text-surface-500 uppercase tracking-wider">Org#</th>
                  <th className="px-2 py-2 text-right text-[10px] text-surface-500 uppercase tracking-wider">Value</th>
                  {prospect.is_pitcher ? (
                    <>
                      <th className="px-2 py-2 text-center text-[10px] text-surface-500 uppercase tracking-wider">FB</th>
                      <th className="px-2 py-2 text-center text-[10px] text-surface-500 uppercase tracking-wider">SL</th>
                      <th className="px-2 py-2 text-center text-[10px] text-surface-500 uppercase tracking-wider">CB</th>
                      <th className="px-2 py-2 text-center text-[10px] text-surface-500 uppercase tracking-wider">CH</th>
                      <th className="px-2 py-2 text-center text-[10px] text-surface-500 uppercase tracking-wider">CMD</th>
                    </>
                  ) : (
                    <>
                      <th className="px-2 py-2 text-center text-[10px] text-surface-500 uppercase tracking-wider">Hit</th>
                      <th className="px-2 py-2 text-center text-[10px] text-surface-500 uppercase tracking-wider">Game</th>
                      <th className="px-2 py-2 text-center text-[10px] text-surface-500 uppercase tracking-wider">Raw</th>
                      <th className="px-2 py-2 text-center text-[10px] text-surface-500 uppercase tracking-wider">Spd</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {prospect.history.map((h, i) => (
                  <tr
                    key={h.year}
                    className={`text-[11px] border-b border-white/[0.03] ${i % 2 === 0 ? 'bg-transparent' : 'bg-white/[0.015]'}`}
                  >
                    <td className="px-2 py-1.5 text-surface-200 font-medium">{h.year}</td>
                    <td className="px-2 py-1.5 text-surface-400">{h.age ? Math.floor(h.age) : '-'}</td>
                    <td className="px-2 py-1.5 text-surface-400">{h.org}</td>
                    <td className="px-2 py-1.5 text-surface-400">{h.position}</td>
                    <td className={`px-2 py-1.5 font-semibold ${fvColor(h.fv)}`}>{h.fv}</td>
                    <td className="px-2 py-1.5 text-center font-mono text-amber-400">{h.top_100 ? '#' + h.top_100 : '-'}</td>
                    <td className="px-2 py-1.5 text-center font-mono text-surface-400">{h.org_rank ? '#' + h.org_rank : '-'}</td>
                    <td className="px-2 py-1.5 text-surface-300 text-right font-mono">{formatValue(h.value)}</td>
                    {prospect.is_pitcher ? (
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

      {/* ── Minor League Statistics ──────────────────────────────── */}
      {milbStats && (milbStats.hitting.length > 0 || milbStats.pitching.length > 0) && (
        <div className="mt-8">
          <h2 className="text-[13px] font-semibold text-surface-300 border-b border-white/[0.06] pb-2 mb-4">
            Minor League Statistics
          </h2>
          <MiLBStatsTable stats={milbStats} expandLatest />
        </div>
      )}
    </div>
  );
}
