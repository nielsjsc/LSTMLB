import React, { useEffect, useState } from 'react';

interface TradeMeterProps {
  team1Name: string;
  team2Name: string;
  differential: number;
}


const TradeMeter: React.FC<TradeMeterProps> = ({ team1Name, team2Name, differential }) => {
  const [angle, setAngle] = useState(0);
  const tickValues = [
    { angle: -45, value: '-$50M' },
    { angle: -22.5, value: '-$25M' },
    { angle: 0, value: '$0' },
    { angle: 22.5, value: '$25M' },
    { angle: 45, value: '$50M' }
  ];
  useEffect(() => {
    const maxDifferential = 50000000;
    // Flip the angle calculation for correct team direction
    const calculatedAngle = Math.min(
      Math.max((-differential / maxDifferential) * 45, -45),
      45
    );
    setAngle(calculatedAngle);
  }, [differential]);

  return (
    <div className="flex flex-col items-center mt-4">
      <div className="relative w-64 h-48">
        <svg className="w-full h-full" viewBox="0 0 100 100">
          {/* Define gradient for meter */}
          <defs>
            <linearGradient id="meterGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="50%" stopColor="#22c55e" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
            <marker
              id="arrowhead"
              markerWidth="8"    // Reduced from 10
              markerHeight="5"   // Reduced from 7
              refX="8"          // Adjusted for new size
              refY="2.5"        // Adjusted for new size
              orient="auto"
            >
              <polygon points="0 0, 8 2.5, 0 5" fill="#dc2626" />
            </marker>
          </defs>
          
          <path
            d="M10,50 A40,40 0 0,1 90,50"
            fill="none"
            stroke="url(#meterGradient)"
            strokeWidth="4"
            className="drop-shadow-md"
          />
          
          {/* Tick marks and values separated */}
          <g className="meter-ticks">
            {tickValues.map(({ angle, value }) => (
              <g key={angle}>
                {/* Tick mark */}
                <line
                  transform={`rotate(${angle}, 50, 50)`}
                  x1="50"
                  y1="15"
                  x2="50"
                  y2="18"
                  stroke="#6b7280"
                  strokeWidth="1"
                />
                {/* Value label - moved outside */}
                <text
                  transform={`
                    rotate(${angle} 50 50)
                    translate(50 4)
                    rotate(${-angle})
                  `}
                  textAnchor="middle"
                  fill="#6b7280"
                  style={{ fontSize: '0.35em' }}
                >
                  {value}
                </text>
              </g>
            ))}
          </g>
          
          {/* Fair trade indicator */}
          <line
            x1="50"
            y1="15"
            x2="50"
            y2="20"
            stroke="#22c55e"
            strokeWidth="2"
          />
          
          {/* Team labels with background - Switched positions */}
          <g className="team-labels">
            <rect x="2" y="42" width="20" height="10" fill="white" rx="2" />
            <text 
              x="5" 
              y="50" 
              className="text-xs font-medium"
              fill="#374151"
              style={{ fontSize: '0.65em' }}
            >
              {team1Name}
            </text>
            
            <rect x="78" y="42" width="20" height="10" fill="white" rx="2" />
            <text 
              x="80" 
              y="50" 
              className="text-xs font-medium"
              fill="#374151"
              style={{ fontSize: '0.65em' }}
            >
              {team2Name}
            </text>
          </g>
          
          {/* Needle with slimmer arrow */}
          <g 
            transform={`rotate(${angle}, 50, 50)`}
            className="transition-all duration-1000 ease-in-out"
          >
            <line
              x1="50"
              y1="50"
              x2="50"
              y2="20"
              stroke="#dc2626"
              strokeWidth="1.5"
              strokeLinecap="round"
              markerEnd="url(#arrowhead)"
            />
            <circle 
              cx="50" 
              cy="50" 
              r="2.5" 
              fill="#dc2626" 
              className="drop-shadow-sm" 
            />
          </g>
        </svg>
      </div>
    </div>
  );
};

export default TradeMeter;