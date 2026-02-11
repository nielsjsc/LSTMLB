/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        brand: {
          50: '#edfcf5',
          100: '#d4f7e6',
          200: '#aceed2',
          300: '#76dfb7',
          400: '#34d399',  // Primary accent
          500: '#15b881',
          600: '#099568',
          700: '#077755',
          800: '#095e45',
          900: '#084d3a',
        },
        surface: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          850: '#172033',
          900: '#0f172a',  // Main background
          950: '#0a0f1e',
        },
        accent: {
          blue: '#60a5fa',
          indigo: '#818cf8',
          sky: '#38bdf8',
        },
        mlb: {
          blue: '#002D72',
          red: '#E31937',
          navy: '#0C2340',
        }
      },
      boxShadow: {
        'glow': '0 0 20px rgba(52, 211, 153, 0.15)',
        'glow-lg': '0 0 40px rgba(52, 211, 153, 0.2)',
        'inner-glow': 'inset 0 1px 0 rgba(255,255,255,0.05)',
      },
      backgroundImage: {
        'gradient-brand': 'linear-gradient(135deg, #34d399, #60a5fa)',
        'gradient-surface': 'linear-gradient(180deg, #1e293b, #0f172a)',
      }
    },
  },
  plugins: [],
}