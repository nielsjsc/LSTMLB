import { motion } from 'framer-motion';
import React, { useState } from 'react';

interface NetworkProps {
  className?: string;
}

const NeuralNetworkViz: React.FC<NetworkProps> = ({ className }) => {
  const [step, setStep] = useState(0);
  const LAYOUT = {
    startX: 100,        // X position of first layer
    layerSpacing: 200,  // Horizontal space between layers
    nodeSpacing: 50,    // Vertical space between nodes
    nodeSize: 14,       // Size of each node
    startY: 50         // Starting Y position
  };
  const stats = [
    { label: 'AVG', value: 0.300 },
    { label: 'OBP', value: 0.380 },
    { label: 'SLG', value: 0.450 },
    { label: 'wOBA', value: 0.350 },
    { label: 'wRC+', value: 120 },
    { label: 'BB%', value: 0.100 },
    { label: 'K%', value: 0.200 },
    { label: 'BABIP', value: 0.320 },
    { label: 'Age', value: 27 }
  ];

  const steps = [
    { title: "Input Layer", description: "Historical baseball statistics enter the network" },
    { title: "LSTM Layers", description: "Two bidirectional LSTM layers process temporal patterns" },
    { title: "Attention", description: "Multi-head attention weighs important time steps" },
    { title: "Output Layer", description: "Predicts next season's statistics" }
  ];
  // SVG path for connections
  const ConnectionLine = ({ start, end, progress }) => (
    <motion.path
      d={`M ${start.x} ${start.y} C ${(start.x + end.x) / 2} ${start.y}, ${(start.x + end.x) / 2} ${end.y}, ${end.x} ${end.y}`}
      stroke="#94a3b8"
      strokeWidth="2"
      fill="none"
      initial={{ pathLength: 0 }}
      animate={{ pathLength: progress }}
      transition={{ duration: 0.5 }}
    />
  );

  return (
    <div className="space-y-4">
      <div className={`relative h-[600px] ${className}`}>
        <svg className="absolute inset-0 w-full h-full">
          <g className="transform translate-x-8 translate-y-8">
            {step >= 1 && stats.map((_, i) => (
              [0,1,2,3].map((j) => (
                <ConnectionLine 
                  key={`${i}-${j}`}
                  start={{ x: LAYOUT.startX, y: LAYOUT.startY + (i * LAYOUT.nodeSpacing) }}
                  end={{ x: LAYOUT.startX + LAYOUT.layerSpacing, y: LAYOUT.startY + 100 + (j * LAYOUT.nodeSpacing * 1.5) }}
                  progress={step >= 1 ? 1 : 0}
                />
              ))
            ))}
            {step >= 2 && [0,1,2,3].map((i) => (
              <ConnectionLine
                key={`lstm-attn-${i}`}
                start={{ x: LAYOUT.startX + LAYOUT.layerSpacing, y: LAYOUT.startY + 100 + (i * LAYOUT.nodeSpacing * 1.5) }}
                end={{ x: LAYOUT.startX + (LAYOUT.layerSpacing * 2), y: LAYOUT.startY + 200 }}
                progress={step >= 2 ? 1 : 0}
              />
            ))}
            {step >= 3 && stats.map((_, i) => (
              <ConnectionLine
                key={`attn-out-${i}`}
                start={{ x: LAYOUT.startX + (LAYOUT.layerSpacing * 2), y: LAYOUT.startY + 200 }}
                end={{ x: LAYOUT.startX + (LAYOUT.layerSpacing * 3), y: LAYOUT.startY + (i * LAYOUT.nodeSpacing) }}
                progress={step >= 3 ? 1 : 0}
              />
            ))}
          </g>
        </svg>
  
        <div className="relative h-full">
          {/* Input Layer */}
          <motion.div 
            className="absolute z-10"
            style={{
              left: LAYOUT.startX,
              top: LAYOUT.startY
            }}
          >
            {stats.map((stat, i) => (
              <motion.div
                key={stat.label}
                className="w-14 h-14 mb-3 bg-blue-500 rounded-full flex items-center justify-center"
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: step >= 0 ? 1 : 0, x: 0 }}
                transition={{ delay: i * 0.1 }}
              >
                <span className="text-white text-xs">{stat.label}</span>
              </motion.div>
            ))}
          </motion.div>
  
          {/* LSTM Layer */}
          <motion.div 
            className="absolute z-10"
            style={{
              left: LAYOUT.startX + LAYOUT.layerSpacing,
              top: LAYOUT.startY + 100
            }}
          >
            {[1,2,3,4].map(i => (
              <motion.div
                key={i}
                className="w-16 h-16 mb-8 bg-purple-500 rounded-full flex items-center justify-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: step >= 1 ? 1 : 0 }}
                transition={{ delay: 0.3 }}
              >
                <span className="text-white text-xs">LSTM</span>
              </motion.div>
            ))}
          </motion.div>
  
          {/* Attention Layer */}
          <motion.div 
            className="absolute z-10"
            style={{
              left: LAYOUT.startX + (LAYOUT.layerSpacing * 2),
              top: LAYOUT.startY + 200
            }}
          >
            <motion.div
              className="w-20 h-20 bg-yellow-500 rounded-full flex items-center justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: step >= 2 ? 1 : 0 }}
            >
              <span className="text-white text-xs">Attention</span>
            </motion.div>
          </motion.div>
  
          {/* Output Layer */}
          <motion.div 
            className="absolute z-10"
            style={{
              left: LAYOUT.startX + (LAYOUT.layerSpacing * 3),
              top: LAYOUT.startY
            }}
          >
            {stats.map((stat, i) => (
              <motion.div
                key={`output-${stat.label}`}
                className="w-14 h-14 mb-3 bg-green-500 rounded-full flex items-center justify-center"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: step >= 3 ? 1 : 0, x: 0 }}
                transition={{ delay: i * 0.1 }}
              >
                <span className="text-white text-xs">{stat.label}</span>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
  
      {/* Step Description */}
      <div className="text-center mb-4">
        <h3 className="text-lg font-semibold">{steps[step].title}</h3>
        <p className="text-gray-600 dark:text-gray-300">{steps[step].description}</p>
      </div>
  
      {/* Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
          disabled={step === 0}
        >
          Previous
        </button>
        <button
          onClick={() => setStep(Math.min(3, step + 1))}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
          disabled={step === 3}
        >
          Next
        </button>
      </div>
    </div>
  );
}

export default NeuralNetworkViz;