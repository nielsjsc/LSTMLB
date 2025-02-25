# LongBall Analytics - MLB Trade Value Calculator

This project started as my attempt to create a trade value metric for MLB players - something I've always been curious about but isn't publicly available. What do we need for trade value? Pretty simple really: projected future WAR and salary data. Basically, how much on-field value will you produce, and how much are you getting paid to do it.

## The Challenge

While we have great projection systems like ZiPS and Steamer that give us 1-3 year forecasts on FanGraphs, they don't tell us how much value Juan Soto will produce in 2035. That's where this project comes in.

## The Solution

I decided to build my own projection models using LSTMs (Long Short-Term Memory networks). Yeah, it's pretty unorthodox in baseball projections, but with the AI boom happening everywhere else, why not give it a shot? Baseball stats are naturally sequential data anyway.

### What It Does

- Projects player performance 15 years into the future
- Calculates trade values based on projected WAR and salary
- Provides a fun trade simulator to test different deals
- Visualizes projections and values through an interactive web interface

## Important Disclaimers

Look, I'm going to be straight up here:

- The hitting projections seem pretty solid
- The pitching projections... not so much (probably due to smaller sample sizes)
- Players with limited MLB experience get wonky projections
- Take everything with a very large, very flaky grain of salt

## Methodology Notes

I made some choices you should know about:

### Playing Time Normalization
- Hitters: 150 games
- Catchers: 135 games
- Starters: 32 games
- Relievers: 65 innings

### WAR to Dollar Calculations
I tweaked the standard linear WAR-to-dollar relationship because I don't think two 2 WAR players equal one 4 WAR player (you get that extra roster spot with the 4 WAR guy).

## Tech Stack

- Backend: Python (FastAPI)
- Frontend: React/TypeScript
- ML: PyTorch (LSTM models)
- Deployment: Render (API) & Netlify (Frontend)

## Local Development

```bash
# Clone the repo
git clone https://github.com/yourusername/longball-analytics.git

# Backend setup
cd web-app/backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend setup
cd ../frontend
npm install
npm run dev
```

## Want to Help?

All the code is here and the models aren't super computationally expensive to train. Feel free to clone the repo and experiment! I'd love to see some improvements to the models.

## Live Demo

Check it out: [LongBall Analytics](https://longball-analytics.netlify.app)

## License

MIT - Go wild with it

---

Remember: Don't take the projections too seriously. If you see it saying "Jackson Chourio will put up 7.1 WAR in 2029", that's just the model doing its best to minimize loss. It's a fun tool for exploring possibilities, not a crystal ball!
