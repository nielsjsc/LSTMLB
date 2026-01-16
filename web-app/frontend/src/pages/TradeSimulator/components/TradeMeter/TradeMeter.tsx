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
          {/* Update gradient colors for dark theme */}
          <defs>
            <linearGradient id="meterGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="50%" stopColor="#10b981" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
            <marker id="arrowhead" markerWidth="8" markerHeight="5" refX="8" refY="2.5" orient="auto">
              <polygon points="0 0, 8 2.5, 0 5" fill="#ef4444" />
            </marker>
          </defs>
          
          <path
            d="M10,50 A40,40 0 0,1 90,50"
            fill="none"
            stroke="url(#meterGradient)"
            strokeWidth="4"
          />
          
          {/* Update tick colors */}
          <g className="meter-ticks">
            {tickValues.map(({ angle, value }) => (
              <g key={angle}>
                <line
                  transform={`rotate(${angle}, 50, 50)`}
                  x1="50"
                  y1="15"
                  x2="50"
                  y2="18"
                  stroke="#4b5563"
                  strokeWidth="1"
                />
                <text
                  transform={`rotate(${angle} 50 50) translate(50 4) rotate(${-angle})`}
                  textAnchor="middle"
                  fill="#9ca3af"
                  style={{ fontSize: '0.35em' }}
                >
                  {value}
                </text>
              </g>
            ))}
          </g>
          
          {/* Update fair trade indicator */}
          <line x1="50" y1="15" x2="50" y2="20" stroke="#10b981" strokeWidth="2" />
          
          {/* Update team labels */}
          <g className="team-labels">
            <text 
              x="20" 
              y="50" 
              fill="#ffffff"
              style={{ fontSize: '0.5em' }}
              textAnchor="start"
              dominantBaseline="middle"
            >
              {team1Name}
            </text>
            
            <text 
              x="80" 
              y="50" 
              fill="#ffffff"
              style={{ fontSize: '0.5em' }}
              textAnchor="end"
              dominantBaseline="middle"
            >
              {team2Name}
            </text>
          </g>
          
          {/* Update needle colors */}
          <g transform={`rotate(${angle}, 50, 50)`} className="transition-all duration-1000 ease-in-out">
            <line
              x1="50"
              y1="50"
              x2="50"
              y2="20"
              stroke="#ef4444"
              strokeWidth="1.5"
              strokeLinecap="round"
              markerEnd="url(#arrowhead)"
            />
            <circle cx="50" cy="50" r="2.5" fill="#ef4444" />
          </g>
        </svg>
      </div>
    </div>
  );
};

export default TradeMeter;