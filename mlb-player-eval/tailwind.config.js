/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        mlb: {
          blue: '#002D72',
          red: '#E31937',
          navy: '#0C2340',
        }
      }
    },
  },
  plugins: [],
}