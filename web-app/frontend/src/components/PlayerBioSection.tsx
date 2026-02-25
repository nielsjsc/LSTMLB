import React from 'react';
import type { PlayerInfo } from '../services/api';

// ── Helpers ──────────────────────────────────────────────────
function formatBirthDate(dateStr: string | null): string {
  if (!dateStr) return '';
  const d = new Date(dateStr + 'T00:00:00');
  return d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
}

function formatBirthplace(bio: PlayerInfo['bio']): string {
  const parts = [bio.birthCity, bio.birthStateProvince, bio.birthCountry].filter(Boolean);
  return parts.join(', ');
}

function formatDebutDate(dateStr: string | null): string {
  if (!dateStr) return '';
  const d = new Date(dateStr + 'T00:00:00');
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatDraft(draft: PlayerInfo['draft']): string {
  if (!draft) return '';
  const parts: string[] = [];
  if (draft.year) parts.push(draft.year);
  if (draft.round && draft.pickNumber) {
    parts.push(`Rd ${draft.round}, Pick ${draft.pickNumber}`);
  }
  if (draft.team) parts.push(draft.team);
  return parts.join(' / ');
}

// Group awards by name, collecting seasons
function groupAwards(awards: PlayerInfo['awards']): { name: string; seasons: string[] }[] {
  const map = new Map<string, string[]>();
  for (const a of awards) {
    if (!map.has(a.name)) map.set(a.name, []);
    map.get(a.name)!.push(a.season);
  }
  // Sort: most occurrences first, then alphabetical
  return Array.from(map.entries())
    .sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]))
    .map(([name, seasons]) => ({ name, seasons: seasons.sort() }));
}

// ── Info pill ────────────────────────────────────────────────
const InfoPill: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="flex flex-col">
    <span className="text-[10px] uppercase tracking-widest text-surface-500">{label}</span>
    <span className="text-sm text-surface-200 font-medium">{value}</span>
  </div>
);

// ── Main Component ───────────────────────────────────────────
interface PlayerBioSectionProps {
  info: PlayerInfo;
  teamColor: string;
}

const PlayerBioSection: React.FC<PlayerBioSectionProps> = ({ info, teamColor }) => {
  const { bio, awards, draft } = info;
  const grouped = groupAwards(awards);
  const birthplace = formatBirthplace(bio);
  const draftStr = formatDraft(draft);

  return (
    <div className="space-y-5">
      {/* Bio details grid */}
      <div className="flex flex-wrap gap-x-8 gap-y-3">
        {bio.height && bio.weight && (
          <InfoPill label="Build" value={`${bio.height}, ${bio.weight} lbs`} />
        )}
        {bio.batSide && bio.pitchHand && (
          <InfoPill label="Bats / Throws" value={`${bio.batSide} / ${bio.pitchHand}`} />
        )}
        {bio.birthDate && (
          <InfoPill label="Born" value={formatBirthDate(bio.birthDate)} />
        )}
        {birthplace && (
          <InfoPill label="Birthplace" value={birthplace} />
        )}
        {bio.mlbDebutDate && (
          <InfoPill label="MLB Debut" value={formatDebutDate(bio.mlbDebutDate)} />
        )}
        {draftStr && (
          <InfoPill label="Drafted" value={draftStr} />
        )}
        {!draft && bio.birthCountry && bio.birthCountry !== 'USA' && (
          <InfoPill label="Signed" value="International Free Agent" />
        )}
      </div>

      {/* Awards */}
      {grouped.length > 0 && (
        <div>
          <h4 className="text-[10px] uppercase tracking-widest text-surface-500 mb-2">Awards</h4>
          <div className="flex flex-wrap gap-1.5">
            {grouped.map(({ name, seasons }) => (
              <span
                key={name}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs border"
                style={{
                  backgroundColor: teamColor + '10',
                  borderColor: teamColor + '25',
                  color: teamColor,
                }}
              >
                {seasons.length > 1 && (
                  <span
                    className="inline-flex items-center justify-center w-4 h-4 rounded-full text-[10px] font-bold"
                    style={{ backgroundColor: teamColor + '30' }}
                  >
                    {seasons.length}
                  </span>
                )}
                <span className="font-medium">{name}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PlayerBioSection;
