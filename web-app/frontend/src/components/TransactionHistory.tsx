import React, { Fragment } from 'react';
import { Link } from 'react-router-dom';
import type { Transaction } from '../services/api';
import { getTeamColors } from '../utils/teamColors';

// ── Type badge config ────────────────────────────────────────
const TYPE_CONFIG: Record<string, { label: string; bg: string; text: string }> = {
  TR:  { label: 'Trade',          bg: 'bg-amber-500/15',   text: 'text-amber-400'   },
  SGN: { label: 'Signed',         bg: 'bg-emerald-500/15', text: 'text-emerald-400'  },
  SFA: { label: 'Signed (FA)',    bg: 'bg-emerald-500/15', text: 'text-emerald-400'  },
  DFA: { label: 'DFA',            bg: 'bg-red-500/15',     text: 'text-red-400'      },
  FA:  { label: 'Free Agent',     bg: 'bg-blue-500/15',    text: 'text-blue-400'     },
  CL:  { label: 'Claimed',        bg: 'bg-purple-500/15',  text: 'text-purple-400'   },
  WV:  { label: 'Waiver',         bg: 'bg-orange-500/15',  text: 'text-orange-400'   },
  SC:  { label: 'Status Change',  bg: 'bg-slate-500/15',   text: 'text-slate-400'    },
  SE:  { label: 'Selected',       bg: 'bg-cyan-500/15',    text: 'text-cyan-400'     },
  RET: { label: 'Retired',        bg: 'bg-gray-500/15',    text: 'text-gray-500'     },
  REL: { label: 'Released',       bg: 'bg-red-500/15',     text: 'text-red-400'      },
};

const DEFAULT_TYPE = { label: 'Transaction', bg: 'bg-slate-500/15', text: 'text-slate-400' };

// ── Helpers ──────────────────────────────────────────────────
function formatDate(dateStr: string): string {
  if (!dateStr) return '';
  const d = new Date(dateStr + 'T00:00:00');
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatYear(dateStr: string): string {
  if (!dateStr) return '';
  return dateStr.slice(0, 4);
}

/** Small team color dot */
const TeamDot: React.FC<{ team: string; className?: string }> = ({ team, className = '' }) => {
  const colors = getTeamColors(team);
  return (
    <span
      className={`inline-block w-2.5 h-2.5 rounded-full ring-1 ring-white/10 ${className}`}
      style={{ backgroundColor: colors.primary }}
    />
  );
};

/** Render description with player names as hyperlinks */
function DescriptionWithLinks({
  description,
  linkedPlayers,
}: {
  description: string;
  linkedPlayers: Transaction['linkedPlayers'];
}) {
  if (!linkedPlayers.length) {
    return <span>{description}</span>;
  }

  // Sort by position in string (longest name first to avoid partial matches)
  const sorted = [...linkedPlayers].sort((a, b) => b.name.length - a.name.length);

  // Build a list of segments: text or link
  type Segment = { type: 'text'; value: string } | { type: 'link'; name: string; id: number };
  const segments: Segment[] = [];

  // Find all player name occurrences and their positions
  const markers: { start: number; end: number; name: string; id: number }[] = [];
  for (const p of sorted) {
    let searchFrom = 0;
    while (true) {
      const idx = description.indexOf(p.name, searchFrom);
      if (idx === -1) break;
      // Check for overlap with existing markers
      const overlaps = markers.some(
        (m) => idx < m.end && idx + p.name.length > m.start
      );
      if (!overlaps) {
        markers.push({ start: idx, end: idx + p.name.length, name: p.name, id: p.mlbId });
      }
      searchFrom = idx + 1;
    }
  }

  // Sort markers by position
  markers.sort((a, b) => a.start - b.start);

  let cursor = 0;
  for (const m of markers) {
    if (m.start > cursor) {
      segments.push({ type: 'text', value: description.slice(cursor, m.start) });
    }
    segments.push({ type: 'link', name: m.name, id: m.id });
    cursor = m.end;
  }
  if (cursor < description.length) {
    segments.push({ type: 'text', value: description.slice(cursor) });
  }

  return (
    <span>
      {segments.map((seg, i) =>
        seg.type === 'link' ? (
          <Link
            key={i}
            to={`/players/${seg.id}`}
            className="text-blue-400 hover:text-blue-300 underline underline-offset-2 decoration-blue-400/40 hover:decoration-blue-300/60 transition-colors"
          >
            {seg.name}
          </Link>
        ) : (
          <Fragment key={i}>{seg.value}</Fragment>
        )
      )}
    </span>
  );
}

// ── Main Component ───────────────────────────────────────────
interface TransactionHistoryProps {
  transactions: Transaction[];
  teamColor: string;
}

const TransactionHistory: React.FC<TransactionHistoryProps> = ({ transactions }) => {
  if (!transactions.length) return null;

  // Group transactions by year for visual separation
  const yearGroups: { year: string; items: Transaction[] }[] = [];
  let currentYear = '';
  for (const txn of transactions) {
    const yr = formatYear(txn.date);
    if (yr !== currentYear) {
      yearGroups.push({ year: yr, items: [] });
      currentYear = yr;
    }
    yearGroups[yearGroups.length - 1].items.push(txn);
  }

  return (
    <div className="space-y-1">
      {yearGroups.map((group) => (
        <div key={group.year}>
          {/* Year divider */}
          <div className="flex items-center gap-3 py-3 px-2">
            <span className="text-xs font-bold text-gray-400 tracking-wider">{group.year}</span>
            <div className="flex-1 h-px bg-gray-50" />
          </div>

          {/* Transaction cards */}
          <div className="space-y-2">
            {group.items.map((txn) => {
              const cfg = TYPE_CONFIG[txn.typeCode] ?? DEFAULT_TYPE;
              const isTrade = txn.typeCode === 'TR';

              return (
                <div
                  key={txn.id}
                  className={`
                    relative rounded-lg border transition-colors
                    ${isTrade
                      ? 'border-amber-500/20 bg-amber-500/[0.03] hover:bg-amber-500/[0.06]'
                      : 'border-gray-100 bg-gray-50/50 hover:bg-white/[0.03]'}
                  `}
                >
                  <div className="px-4 py-3">
                    {/* Header row: badge + date + teams */}
                    <div className="flex items-center gap-2 mb-2 flex-wrap">
                      {/* Type badge */}
                      <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-semibold ${cfg.bg} ${cfg.text}`}>
                        {cfg.label}
                      </span>

                      {/* Date */}
                      <span className="text-xs text-gray-500">{formatDate(txn.date)}</span>

                      {/* Team flow arrow (for trades / moves) */}
                      {(txn.fromTeam || txn.toTeam) && (
                        <span className="inline-flex items-center gap-1.5 ml-auto text-xs text-gray-500">
                          {txn.fromTeam && (
                            <span className="inline-flex items-center gap-1">
                              <TeamDot team={txn.fromTeam} />
                              <span>{txn.fromTeam}</span>
                            </span>
                          )}
                          {txn.fromTeam && txn.toTeam && (
                            <svg className="w-3.5 h-3.5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                            </svg>
                          )}
                          {txn.toTeam && (
                            <span className="inline-flex items-center gap-1">
                              <TeamDot team={txn.toTeam} />
                              <span>{txn.toTeam}</span>
                            </span>
                          )}
                        </span>
                      )}
                    </div>

                    {/* Description */}
                    <p className="text-sm text-gray-600 leading-relaxed">
                      <DescriptionWithLinks
                        description={txn.description}
                        linkedPlayers={txn.linkedPlayers}
                      />
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
};

export default TransactionHistory;
