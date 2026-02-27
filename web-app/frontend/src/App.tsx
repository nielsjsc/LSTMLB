import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home/Home'
import TradeSimulator from './pages/TradeSimulator/TradeSimulator'
import PlayerDetails from './pages/PlayerDetails/PlayerDetails'
import Projections from './pages/Projections/Projections'
import Prospects from './pages/Prospects/Prospects'
import ProspectDetail from './pages/Prospects/ProspectDetail'
import About from './pages/About/About'
import TradeValues from './pages/TradeValues/TradeValues'
import PastTrades from './pages/PastTrades/PastTrades'
import TradeDetail from './pages/TradeDetail/TradeDetail'
import NotFound from './pages/NotFound/NotFound'

function App() {
  return (
    <Router>
        <div className="min-h-screen w-full bg-surface-900">
        <Navbar />
        <div className="px-4 pt-20">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/tradesimulator" element={<TradeSimulator />} />
            <Route path="/players/:playerId" element={<PlayerDetails />} />
            <Route path="/projections" element={<Projections />} />
            <Route path="/prospects" element={<Prospects />} />
            <Route path="/prospects/:prospectId" element={<ProspectDetail />} />
            <Route path="/about" element={<About />} />
            <Route path="/tradevalues" element={<TradeValues />} />
            <Route path="/trades" element={<PastTrades />} />
            <Route path="/trades/:tradeId" element={<TradeDetail />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </div>
      </div>
    </Router>
  )
}

export default App