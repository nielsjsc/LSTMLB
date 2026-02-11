import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { getPlayerDetails, PlayerStats } from '../../services/api';
import { CURRENT_YEAR, MAX_PROJECTION_YEARS, API_BASE } from '../../config';
import { CombinedHittingTable, CombinedPitchingTable } from '../../components/Tables';
import { getTeamColors, getTeamName } from '../../utils/teamColors';

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

// ─── Subcomponents ──────────────────────────────────────────

/** Headshot with fallback silhouette */
const PlayerHeadshot: React.FC<{ mlbId: number | null; name: string; teamColor: string }> = ({
  mlbId,
  name,
  teamColor,
}) => {
  const [imgError, setImgError] = useState(false);
  const src = mlbId ? `${API_BASE}/headshots/${mlbId}.png` : null;

  return (
    <div
      className="relative w-28 h-28 md:w-36 md:h-36 rounded-2xl overflow-hidden border-2 shrink-0"
      style={{ borderColor: teamColor + '80' }}
    >
      {src && !imgError ? (
        <img
          src={src}
          alt={name}
          className="w-full h-full object-cover"
          onError={() => setImgError(true)}
        />
      ) : (
        <div className="w-full h-full bg-surface-700 flex items-center justify-center">
          <svg className="w-16 h-16 text-surface-500" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z" />
          </svg>
        </div>
      )}
    </div>
  );
};

/** Small key-value stat used in the overview grid */
const MiniStat: React.FC<{
  label: string;
  value: string;
  highlight?: boolean;
  negative?: boolean;
  teamAccent?: string;
}> = ({ label, value, highlight, negative, teamAccent }) => (
  <div className="flex flex-col">
    <span className="text-[11px] uppercase tracking-wider text-surface-500 mb-0.5">{label}</span>
    <span
      className={`text-lg font-semibold leading-tight ${
        negative ? 'text-red-400' : highlight ? '' : 'text-white'
      }`}
      style={highlight && !negative ? { color: teamAccent } : undefined}
    >
      {value}
    </span>
  </div>
);

/** Value card (trade value, surplus, contract, etc.) */
const ValueCard: React.FC<{
  title: string;
  items: Array<{ label: string; value: string; positive?: boolean }>;
  teamColor: string;
}> = ({ title, items, teamColor }) => (
  <div className="rounded-xl p-5 border border-white/[0.06] bg-surface-800/60">
    <h3
      className="text-xs font-semibold uppercase tracking-widest mb-4"
      style={{ color: teamColor }}
    >
      {title}
    </h3>
    <div className="space-y-3">
      {items.map((item) => (
        <div key={item.label} className="flex justify-between items-baseline">
          <span className="text-sm text-surface-400">{item.label}</span>
          <span
            className={`text-base font-semibold ${
              item.positive === false ? 'text-red-400' : item.positive ? '' : 'text-white'
            }`}
            style={item.positive ? { color: teamColor } : undefined}
          >
            {item.value}
          </span>
        </div>
      ))}
    </div>
  </div>
);

// ─── Main Component ─────────────────────────────────────────

const PlayerDetails: React.FC = () => {
  const { playerId } = useParams<{ playerId: string }>();
  const [player, setPlayer] = useState<PlayerStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  // ── Derived state ──
  const cur = player?.projections.find((p) => p.year === CURRENT_YEAR) ?? player?.projections[0];
  const team = cur?.team ?? player?.team ?? '';
  const colors = getTeamColors(team);
  const teamName = getTeamName(team);

  const MAX_PROJECTION_YEAR = CURRENT_YEAR + MAX_PROJECTION_YEARS - 1;

  const hasPitching = player?.projections.some((p) => p.pitching?.war_pit != null);
  const hasHitting = player?.projections.some((p) => p.hitting?.war_bat != null);

  const hasCurrentHitting = player?.projections.some(
    (p) => p.year === CURRENT_YEAR && p.hitting?.war_bat != null
  );
  const hasCurrentPitching = player?.projections.some(
    (p) => p.year === CURRENT_YEAR && p.pitching?.war_pit != null
  );

  const pitchingTableData = React.useMemo(() => {
    if (!player?.projections) return [];
    return player.projections
      .filter(
        (proj): proj is typeof proj & { pitching: NonNullable<typeof proj.pitching> } =>
          proj.pitching?.war_pit != null && proj.year <= MAX_PROJECTION_YEAR
      )
      .map((proj) => ({ year: proj.year, age: proj.age, status: proj.status, value: proj.value, pitching: proj.pitching }));
  }, [player, MAX_PROJECTION_YEAR]);

  const hittingTableData = React.useMemo(() => {
    if (!player?.projections) return [];
    return player.projections
      .filter(
        (proj): proj is typeof proj & { hitting: NonNullable<typeof proj.hitting> } =>
          proj.hitting?.war_bat != null && proj.year <= MAX_PROJECTION_YEAR
      )
      .map((proj) => ({ year: proj.year, age: proj.age, status: proj.status, value: proj.value, hitting: proj.hitting }));
  }, [player, MAX_PROJECTION_YEAR]);

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

  // ── FA year display ──
  const faEarliest = cur?.earliest_fa_year;
  const faProbable = cur?.probable_fa_year;
  const faLatest = cur?.fa_year;

  // Hitting overview stats
  const h = cur?.hitting;
  const p = cur?.pitching;
  const v = cur?.value;

  return (
    <div className="min-h-screen bg-surface-900">
      {/* ════════ HERO HEADER ════════ */}
      <div className="relative overflow-hidden">
        {/* Team gradient background */}
        <div className="absolute inset-0" style={{ background: colors.gradient, opacity: 0.15 }} />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-surface-900/60 to-surface-900" />

        <div className="relative max-w-6xl mx-auto px-4 pt-8 pb-10 md:pt-12 md:pb-14">
          <div className="flex items-start gap-6 md:gap-8">
            {/* Headshot */}
            <PlayerHeadshot
              mlbId={player.mlb_id}
              name={player.name}
              teamColor={colors.primary}
            />

            {/* Identity */}
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2 mb-1">
                <span
                  className="text-xs font-bold uppercase tracking-widest"
                  style={{ color: colors.accent }}
                >
                  {teamName}
                </span>
              </div>

              <h1 className="text-3xl md:text-4xl font-extrabold text-white tracking-tight mb-3 truncate">
                {player.name}
              </h1>

              {/* Badges */}
              <div className="flex flex-wrap gap-2 mb-5">
                <span
                  className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide"
                  style={{ backgroundColor: colors.primary + '20', color: colors.accent }}
                >
                  {player.position}
                </span>
                {cur?.age && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-surface-700/60 text-surface-300">
                    Age {cur.age}
                  </span>
                )}
                {cur?.status && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-surface-700/60 text-surface-300">
                    {cur.status}
                  </span>
                )}
                {faProbable && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-surface-700/60 text-surface-300">
                    FA: {faProbable}
                  </span>
                )}
              </div>

              {/* Key Stat Overview */}
              <div className="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-7 gap-x-6 gap-y-3">
                {hasHitting && h && (
                  <>
                    <MiniStat label="WAR" value={fmt.war(h.war_bat)} highlight teamAccent={colors.accent} />
                    <MiniStat label="AVG" value={fmt.dec(h.avg)} />
                    <MiniStat label="OPS" value={fmt.dec(h.ops)} />
                    <MiniStat label="wRC+" value={fmt.int(h.wrc_plus)} highlight teamAccent={colors.accent} />
                    <MiniStat label="HR" value={fmt.int(h.hr)} />
                    <MiniStat label="SB" value={fmt.int(h.sb)} />
                    <MiniStat label="wOBA" value={fmt.dec(h.woba)} />
                  </>
                )}
                {hasPitching && p && !hasHitting && (
                  <>
                    <MiniStat label="WAR" value={fmt.war(p.war_pit)} highlight teamAccent={colors.accent} />
                    <MiniStat label="ERA" value={fmt.dec(p.era)} />
                    <MiniStat label="FIP" value={fmt.dec(p.fip)} />
                    <MiniStat label="SIERA" value={fmt.dec(p.siera)} />
                    <MiniStat label="K%" value={fmt.pct(p.k_pct_pit)} highlight teamAccent={colors.accent} />
                    <MiniStat label="BB%" value={fmt.pct(p.bb_pct_pit)} />
                    <MiniStat label="GS" value={fmt.int(p.gs)} />
                  </>
                )}
                {hasPitching && p && hasHitting && (
                  <>
                    <MiniStat label="WAR (bat)" value={fmt.war(h?.war_bat)} highlight teamAccent={colors.accent} />
                    <MiniStat label="WAR (pit)" value={fmt.war(p.war_pit)} highlight teamAccent={colors.accent} />
                    <MiniStat label="ERA" value={fmt.dec(p.era)} />
                    <MiniStat label="AVG" value={fmt.dec(h?.avg)} />
                    <MiniStat label="OPS" value={fmt.dec(h?.ops)} />
                    <MiniStat label="K%" value={fmt.pct(p.k_pct_pit)} />
                    <MiniStat label="wRC+" value={fmt.int(h?.wrc_plus)} />
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ════════ VALUE CARDS ════════ */}
      <div className="max-w-6xl mx-auto px-4 -mt-2 mb-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <ValueCard
            title="Trade Value"
            teamColor={colors.accent}
            items={[
              { label: 'Surplus', value: fmt.dollar(v?.trade_value), positive: (v?.trade_value ?? 0) >= 0 },
              { label: 'Years Ctrl', value: v?.years_control != null ? `${v.years_control}` : '—' },
              { label: 'Ctrl Through', value: v?.control_through != null ? `${v.control_through}` : '—' },
            ]}
          />
          <ValueCard
            title="Production"
            teamColor={colors.accent}
            items={[
              { label: 'Proj WAR', value: fmt.war(v?.contract_war) },
              { label: 'Proj Value', value: fmt.dollar(v?.contract_base_value), positive: true },
              { label: 'Avg WAR/yr', value: fmt.war(v?.avg_war) },
            ]}
          />
          <ValueCard
            title="Contract"
            teamColor={colors.accent}
            items={[
              { label: 'Total $', value: fmt.dollar(v?.total_contract), positive: false },
              { label: 'Avg $/yr', value: fmt.dollar(v?.avg_contract), positive: false },
              {
                label: 'FA Window',
                value:
                  faEarliest && faLatest
                    ? faEarliest === faLatest
                      ? `${faLatest}`
                      : `${faEarliest}–${faLatest}`
                    : '—',
              },
            ]}
          />
          <ValueCard
            title="Career / History"
            teamColor={colors.accent}
            items={[
              { label: 'Hist WAR', value: fmt.war(v?.historical_war) },
              { label: 'Hist Value', value: fmt.dollar(v?.historical_value), positive: true },
              { label: 'Total WAR', value: fmt.war(v?.total_war) },
            ]}
          />
        </div>
      </div>

      {/* ════════ STAT TABLES ════════ */}
      <div className="max-w-6xl mx-auto px-4 pb-16 space-y-6">
        {hasPitching && hasCurrentPitching && (
          <section className="rounded-xl overflow-hidden border border-white/[0.06] bg-surface-800/40">
            <div className="px-6 py-4 border-b border-white/[0.06] flex items-center gap-3">
              <div className="w-1 h-5 rounded-full" style={{ backgroundColor: colors.primary }} />
              <h2 className="text-lg font-semibold text-white">Pitching Projections</h2>
            </div>
            <CombinedPitchingTable data={pitchingTableData.sort((a, b) => b.year - a.year)} />
          </section>
        )}

        {hasHitting && hasCurrentHitting && (
          <section className="rounded-xl overflow-hidden border border-white/[0.06] bg-surface-800/40">
            <div className="px-6 py-4 border-b border-white/[0.06] flex items-center gap-3">
              <div className="w-1 h-5 rounded-full" style={{ backgroundColor: colors.primary }} />
              <h2 className="text-lg font-semibold text-white">Hitting Projections</h2>
            </div>
            <CombinedHittingTable data={hittingTableData.sort((a, b) => b.year - a.year)} />
          </section>
        )}
      </div>
    </div>
  );
};

export default PlayerDetails;
