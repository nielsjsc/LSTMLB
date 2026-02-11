import React, { useEffect, useState, useMemo } from 'react';

interface TradeMeterProps {
  team1Name: string;
  team2Name: string;
  differential: number;
}

const TradeMeter: React.FC<TradeMeterProps> = ({ team1Name, team2Name, differential }) => {
  const [displayAngle, setDisplayAngle] = useState(0);
  const maxDifferential = 50_000_000;

  // Calculate target angle: negative diff = team1 wins (left), positive = team2 wins (right)
  const targetAngle = useMemo(() => {
    return Math.min(Math.max((-differential / maxDifferential) * 90, -90), 90);
  }, [differential]);

  // Smooth spring animation
  useEffect(() => {
    let frame: number;
    let current = displayAngle;
    const spring = () => {
      const diff = targetAngle - current;
      if (Math.abs(diff) < 0.3) {
        setDisplayAngle(targetAngle);
        return;
      }
      current = current + diff * 0.08;
      setDisplayAngle(current);
      frame = requestAnimationFrame(spring);
    };
    frame = requestAnimationFrame(spring);
    return () => cancelAnimationFrame(frame);
  }, [targetAngle]);

  // Compute verdict
  const absDiff = Math.abs(differential);
  const getVerdict = () => {
    if (absDiff < 2_000_000) return { label: 'FAIR TRADE', color: '#34d399', emoji: '✅' };
    if (absDiff < 10_000_000) return { label: 'SLIGHT EDGE', color: '#a3e635', emoji: '📊' };
    if (absDiff < 25_000_000) return { label: 'UNEVEN', color: '#fbbf24', emoji: '⚠️' };
    if (absDiff < 40_000_000) return { label: 'LOPSIDED', color: '#f97316', emoji: '🔥' };
    return { label: 'ROBBERY', color: '#ef4444', emoji: '🚨' };
  };
  const verdict = getVerdict();

  const formatDollar = (val: number) => {
    if (val >= 1_000_000) return `$${(val / 1_000_000).toFixed(1)}M`;
    if (val >= 1_000) return `$${(val / 1_000).toFixed(0)}K`;
    return `$${val}`;
  };

  // Gauge arc parameters
  const cx = 150, cy = 140, r = 110;
  const startAngle = -180;
  const endAngle = 0;
  
  const polarToCart = (angleDeg: number, radius: number) => {
    const rad = (angleDeg * Math.PI) / 180;
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
  };

  const arcStart = polarToCart(startAngle, r);
  const arcEnd = polarToCart(endAngle, r);

  // Tick marks
  const ticks = [
    { angle: -180, label: `-$50M` },
    { angle: -157.5, label: '' },
    { angle: -135, label: `-$25M` },
    { angle: -112.5, label: '' },
    { angle: -90, label: `$0` },
    { angle: -67.5, label: '' },
    { angle: -45, label: `$25M` },
    { angle: -22.5, label: '' },
    { angle: 0, label: `$50M` },
  ];

  // Needle angle: map displayAngle (-90..90) to arc (-180..0)
  const needleAngleDeg = -90 + (displayAngle / 90) * (-90);
  const needleTip = polarToCart(needleAngleDeg, r - 14);
  const needleBase1 = polarToCart(needleAngleDeg - 90, 5);
  const needleBase2 = polarToCart(needleAngleDeg + 90, 5);

  const needleGlowColor = verdict.color;
  const favoredTeam = differential > 0 ? team1Name : differential < 0 ? team2Name : null;

  return (
    <div className="flex flex-col items-center py-6">
      {/* Gauge */}
      <div className="relative" style={{ width: 300, height: 190 }}>
        <svg viewBox="0 0 300 180" className="w-full h-full overflow-visible">
          <defs>
            <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="20%" stopColor="#f97316" />
              <stop offset="35%" stopColor="#fbbf24" />
              <stop offset="50%" stopColor="#34d399" />
              <stop offset="65%" stopColor="#fbbf24" />
              <stop offset="80%" stopColor="#f97316" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
            <filter id="gaugeGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="3" />
            </filter>
            <filter id="needleGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="4" />
            </filter>
            <radialGradient id="centerGrad" cx="50%" cy="100%" r="60%">
              <stop offset="0%" stopColor="#1e293b" />
              <stop offset="100%" stopColor="#0f172a" />
            </radialGradient>
          </defs>

          {/* Background arc glow */}
          <path
            d={`M ${arcStart.x} ${arcStart.y} A ${r} ${r} 0 0 1 ${arcEnd.x} ${arcEnd.y}`}
            fill="none"
            stroke="url(#gaugeGrad)"
            strokeWidth="14"
            strokeLinecap="round"
            opacity="0.15"
            filter="url(#gaugeGlow)"
          />
          
          {/* Main arc track (dark) */}
          <path
            d={`M ${arcStart.x} ${arcStart.y} A ${r} ${r} 0 0 1 ${arcEnd.x} ${arcEnd.y}`}
            fill="none"
            stroke="#1e293b"
            strokeWidth="10"
            strokeLinecap="round"
          />

          {/* Colored arc */}
          <path
            d={`M ${arcStart.x} ${arcStart.y} A ${r} ${r} 0 0 1 ${arcEnd.x} ${arcEnd.y}`}
            fill="none"
            stroke="url(#gaugeGrad)"
            strokeWidth="10"
            strokeLinecap="round"
            opacity="0.85"
          />

          {/* Zone segment markers */}
          {[-135, -90, -45].map((angle) => {
            const p1 = polarToCart(angle, r - 8);
            const p2 = polarToCart(angle, r + 8);
            return (
              <line key={angle} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
                stroke="#334155" strokeWidth="1.5" />
            );
          })}

          {/* Tick marks and labels */}
          {ticks.map(({ angle, label }) => {
            const outer = polarToCart(angle, r + 14);
            const inner = polarToCart(angle, r + 6);
            const labelPos = polarToCart(angle, r + 28);
            const isMajor = label !== '';
            return (
              <g key={angle}>
                <line x1={inner.x} y1={inner.y} x2={outer.x} y2={outer.y}
                  stroke={isMajor ? '#64748b' : '#334155'} strokeWidth={isMajor ? 1.5 : 1} />
                {label && (
                  <text x={labelPos.x} y={labelPos.y} textAnchor="middle" dominantBaseline="middle"
                    fill="#94a3b8" style={{ fontSize: '9px', fontFamily: 'Inter, sans-serif' }}>
                    {label}
                  </text>
                )}
              </g>
            );
          })}

          {/* Fair trade indicator at top center */}
          <line x1={cx} y1={cy - r - 6} x2={cx} y2={cy - r + 6}
            stroke="#34d399" strokeWidth="2.5" strokeLinecap="round" />

          {/* Needle glow (behind needle) */}
          <polygon
            points={`${needleTip.x},${needleTip.y} ${needleBase1.x},${needleBase1.y} ${needleBase2.x},${needleBase2.y}`}
            fill={needleGlowColor} opacity="0.4" filter="url(#needleGlow)"
          />

          {/* Needle */}
          <polygon
            points={`${needleTip.x},${needleTip.y} ${needleBase1.x},${needleBase1.y} ${needleBase2.x},${needleBase2.y}`}
            fill={needleGlowColor}
            style={{ filter: `drop-shadow(0 0 6px ${needleGlowColor}80)` }}
          />

          {/* Center hub */}
          <circle cx={cx} cy={cy} r="10" fill="url(#centerGrad)" stroke="#334155" strokeWidth="1" />
          <circle cx={cx} cy={cy} r="4" fill={needleGlowColor} opacity="0.9" />
          <circle cx={cx} cy={cy} r="6" fill="none" stroke={needleGlowColor} strokeWidth="0.5" opacity="0.4" />

          {/* Team labels */}
          <text x="28" y={cy + 20} fill="#94a3b8" textAnchor="start"
            style={{ fontSize: '11px', fontWeight: 600, fontFamily: 'Inter, sans-serif', letterSpacing: '0.05em' }}>
            {team1Name}
          </text>
          <text x="272" y={cy + 20} fill="#94a3b8" textAnchor="end"
            style={{ fontSize: '11px', fontWeight: 600, fontFamily: 'Inter, sans-serif', letterSpacing: '0.05em' }}>
            {team2Name}
          </text>
        </svg>
      </div>

      {/* Digital Readout */}
      <div className="flex flex-col items-center mt-2 gap-1.5">
        <div className="flex items-center gap-2">
          <span className="text-lg">{verdict.emoji}</span>
          <span className="text-sm font-bold tracking-widest uppercase"
            style={{ color: verdict.color }}>
            {verdict.label}
          </span>
          <span className="text-lg">{verdict.emoji}</span>
        </div>
        {favoredTeam && absDiff >= 2_000_000 && (
          <p className="text-surface-400 text-xs font-medium">
            <span style={{ color: verdict.color }} className="font-semibold">
              {formatDollar(absDiff)}
            </span>
            {' '}edge for{' '}
            <span className="text-white font-semibold">{favoredTeam}</span>
          </p>
        )}
      </div>
    </div>
  );
};

export default TradeMeter;