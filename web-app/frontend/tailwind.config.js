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
        display: ['Barlow', 'Inter', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'Consolas', 'monospace'],
      },
      colors: {
        brand: {
          50: '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#f59e0b',  // Primary accent — warm amber
          500: '#d97706',
          600: '#b45309',
          700: '#92400e',
          800: '#78350f',
          900: '#451a03',
        },
        surface: {
          50: '#fafafa',
          100: '#f5f5f5',
          200: '#e5e5e5',
          300: '#d4d4d4',
          400: '#a3a3a3',
          500: '#737373',
          600: '#525252',
          700: '#2a2a2a',
          800: '#1a1a1a',
          850: '#151515',
          900: '#111111',  // Main background — true dark
          950: '#0a0a0a',
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
        'glow': '0 0 20px rgba(245, 158, 11, 0.12)',
        'glow-lg': '0 0 40px rgba(245, 158, 11, 0.18)',
        'inner-glow': 'inset 0 1px 0 rgba(255,255,255,0.04)',
      },
      backgroundImage: {
        'gradient-brand': 'linear-gradient(135deg, #f59e0b, #d97706)',
        'gradient-surface': 'linear-gradient(180deg, #1a1a1a, #111111)',
      }
    },
  },
  plugins: [],
}