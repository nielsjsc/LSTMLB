import React from 'react'
import { FaBalanceScale,FaStar, FaDatabase, FaExchangeAlt, FaCode, FaCalculator, FaRunning, FaGithub, FaDollarSign, FaBaseballBall, FaChalkboardTeacher } from 'react-icons/fa'
import {BsListStars } from 'react-icons/bs'
import CollapsibleSection from './CollapsibleSection'



const Card = ({ children, className = "" }: { children: React.ReactNode, className?: string }) => (
  <div className={`bg-surface-800/50 backdrop-blur-sm rounded-2xl 
                   shadow-lg ring-1 ring-white/[0.06] 
                   p-8 transition-all hover:shadow-xl ${className}`}>
    {children}
  </div>
)

const SubHeading = ({ children, icon: Icon }: { children: React.ReactNode, icon?: any }) => (
  <h3 className="flex items-center gap-3 text-xl font-semibold mb-6 
                 text-white tracking-tight">
    {Icon && <Icon className="text-brand-400" />}
    {children}
  </h3>
)

const TechnicalOverview = () => {
  return (
    <section className="relative py-16">
      {/* Gradient background */}
      <div className="absolute inset-0 bg-gradient-to-b from-surface-900 to-surface-800 -z-10" />
      
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Data Collection Section */}
        <CollapsibleSection
          title="Data Collection & Processing"
          subtitle="MLB, MiLB, and salary data collection pipeline"
          icon={FaDatabase}
          defaultOpen={false}
        >
          <div className="space-y-8">
            <Card>
              <SubHeading icon={FaDatabase}>Data Sources</SubHeading>
              <div className="grid md:grid-cols-3 gap-8">
                <div className="space-y-4">
                  <h4 className="font-semibold text-lg text-white">
                    MLB Statistics
                  </h4>
                  <p className="text-surface-300 leading-relaxed">
                    Using <a href="https://github.com/jldbc/pybaseball" 
                      className="text-brand-400 hover:text-emerald-400 font-medium 
                              transition-colors">
                      pybaseball
                    </a>, we collect comprehensive MLB statistics from 2000-present, including
                    traditional stats, advanced metrics, and Statcast data.
                  </p>
                </div>

                <div className="space-y-4">
                  <h4 className="font-semibold text-lg text-white">
                    Minor League Statistics
                  </h4>
                  <p className="text-surface-300 leading-relaxed">
                    Prospect and minor league data is sourced from Fangraphs, providing
                    performance metrics across all levels (A, AA, AAA), including
                    age-adjusted statistics and position information.
                  </p>
                </div>

                <div className="space-y-4">
                  <h4 className="font-semibold text-lg text-white">
                    Salary Data
                  </h4>
                  <p className="text-surface-300 leading-relaxed">
                    Contract information from Spotrac includes current contracts,
                    arbitration status, service time tracking, and historical salary data
                    essential for value projections.
                  </p>
                </div>
              </div>
            </Card>

            <Card>
              <SubHeading icon={FaCode}>Data Pipeline</SubHeading>
              <div className="space-y-6">
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="flex items-start gap-3 p-4 bg-surface-700/50 
                                rounded-xl transition-colors">
                    <FaDatabase className="mt-1 text-brand-400 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-white mb-2">
                        Data Collection
                      </h4>
                      <p className="text-sm text-surface-300">
                        Automated statistics gathering through pybaseball API with
                        comprehensive error handling.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3 p-4 bg-surface-700/50 
                                rounded-xl transition-colors">
                    <FaChalkboardTeacher className="mt-1 text-brand-400 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-white mb-2">
                        Data Processing
                      </h4>
                      <p className="text-sm text-surface-300">
                        Standardization of team abbreviations and position classifications
                        across sources.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3 p-4 bg-surface-700/50 
                                rounded-xl transition-colors">
                    <BsListStars className="mt-1 text-brand-400 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-white mb-2">
                        Data Integration
                      </h4>
                      <p className="text-sm text-surface-300">
                        Careful matching of player records across MLB stats, minor league
                        data, and salary information.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </CollapsibleSection>
      
         
        {/* Model Architecture Section */}
        <CollapsibleSection
          title="Batting Projections Model"
          subtitle="Long-term career trajectory predictions using LSTM networks"
          icon={FaBaseballBall}
          defaultOpen={false}
        >
          <div className="space-y-8">
            <Card>
              <div className="flex justify-between items-center mb-8">
                <SubHeading icon={FaBaseballBall}>Model Overview</SubHeading>
                <div className="flex items-center gap-4">
                  <a href="https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-brand-400 hover:text-emerald-400">
                    Neural Networks Explained
                  </a>
                  <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-brand-400 hover:text-emerald-400">
                    Understanding LSTMs
                  </a>
                  <a href="https://github.com/nielsjsc/LSTMLB/blob/main/models/batter.ipynb"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500 
                            hover:bg-emerald-400 text-white transition-colors text-sm">
                    <FaGithub /> View Code
                  </a>
                </div>
              </div>

              <div className="space-y-8">
                {/* Model Architecture */}
                <div className="grid md:grid-cols-2 gap-8">
                  <div>
                    <h4 className="font-semibold text-lg text-white mb-4">
                      How It Works
                    </h4>
                    <ul className="space-y-3">
                      <li className="flex items-start gap-3 text-surface-300">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0 mt-2" />
                        <p>Takes 4 seasons of historical data to predict future performance</p>
                      </li>
                      <li className="flex items-start gap-3 text-surface-300">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0 mt-2" />
                        <p>Uses rate statistics to account for varying playing time</p>
                      </li>
                      <li className="flex items-start gap-3 text-surface-300">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0 mt-2" />
                        <p>LSTM network learns patterns in career progression</p>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-surface-700/50 rounded-xl p-6">
                    <h4 className="font-semibold text-lg text-white mb-4">
                      Input Features
                    </h4>
                    <div className="grid grid-cols-2 gap-3 text-sm text-surface-300">
                      <div className="space-y-2">
                        <p className="font-medium">Counting Stats (Rate Adjusted)</p>
                        <ul className="space-y-1">
                          <li>Age</li>
                          <li>Home Runs</li>
                          <li>Doubles</li>
                          <li>Triples</li>
                          <li>RBI</li>
                          <li>Runs</li>
                        </ul>
                      </div>
                      <div className="space-y-2">
                        <p className="font-medium">Rate Metrics</p>
                        <ul className="space-y-1">
                          <li>Walk Rate</li>
                          <li>Strikeout Rate</li>
                          <li>wOBA</li>
                          <li>wRC+</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Sliding Window Example */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Sliding Window Prediction Process
                  </h4>
                  <div className="space-y-6">
                    <p className="text-sm text-surface-300">
                      The model uses a 4-year sliding window to make predictions, with each prediction
                      becoming input for future projections.
                    </p>

                    <div className="space-y-4">
                      <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-emerald-500">
                        <h5 className="text-sm font-medium text-white mb-2">
                          2025 Prediction Uses
                        </h5>
                        <div className="grid grid-cols-5 gap-2 text-xs">
                          <div className="col-span-4 space-y-1 text-surface-300">
                            <p>2021: .287/.373/.544, 39 HR</p>
                            <p>2022: .311/.425/.686, 62 HR</p>
                            <p>2023: .267/.406/.613, 37 HR</p>
                            <p>2024: .322/.458/.701, 58 HR</p>
                          </div>
                          <div className="text-emerald-600 dark:text-emerald-400">
                            → 2025: .285/.402/.602, 50 HR
                          </div>
                        </div>
                      </div>

                      <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-blue-500">
                        <h5 className="text-sm font-medium text-white mb-2">
                          2026 Prediction Uses
                        </h5>
                        <div className="grid grid-cols-5 gap-2 text-xs">
                          <div className="col-span-4 space-y-1 text-surface-300">
                            <p>2022: .311/.425/.686, 62 HR</p>
                            <p>2023: .267/.406/.613, 37 HR</p>
                            <p>2024: .322/.458/.701, 58 HR</p>
                            <p className="text-emerald-600">2025: .285/.402/.602, 50 HR</p>
                          </div>
                          <div className="text-emerald-600 dark:text-emerald-400">
                            → 2026: .270/.384/.573, 48 HR
                          </div>
                        </div>
                      </div>

                      <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-purple-500">
                        <h5 className="text-sm font-medium text-white mb-2">
                          2027 Prediction Uses
                        </h5>
                        <div className="grid grid-cols-5 gap-2 text-xs">
                          <div className="col-span-4 space-y-1 text-surface-300">
                            <p>2023: .267/.406/.613, 37 HR</p>
                            <p>2024: .322/.458/.701, 58 HR</p>
                            <p className="text-emerald-600">2025: .285/.402/.602, 50 HR</p>
                            <p className="text-emerald-600">2026: .270/.384/.573, 48 HR</p>
                          </div>
                          <div className="text-emerald-600 dark:text-emerald-400">
                            → 2027: .264/.372/.552, 45 HR
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="text-sm text-surface-300 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                      <span className="font-medium">Note:</span> The LSTM's memory capabilities allow it to learn
                      long-term patterns in player development, while the sliding window approach ensures 
                      predictions incorporate the most recent performance data.
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </CollapsibleSection>
        <CollapsibleSection
          title="Pitching Projections Models"
          subtitle="Split SP/RP prediction system"
          icon={FaBaseballBall}
          defaultOpen={false}
        >
          <div className="space-y-8">
            <Card>
              <div className="flex justify-between items-center mb-8">
                <SubHeading icon={FaBaseballBall}>Model Overview</SubHeading>
                <div className="flex items-center gap-4">
                  <a href="https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-brand-400 hover:text-emerald-400">
                    Neural Networks Explained
                  </a>
                  <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-brand-400 hover:text-emerald-400">
                    Understanding LSTMs
                  </a>
                  <a href="https://github.com/nielsjsc/LSTMLB/blob/main/models/pitcher.ipynb"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500 
                            hover:bg-emerald-400 text-white transition-colors text-sm">
                    <FaGithub /> View Code
                  </a>
                </div>
              </div>

              <div className="space-y-8">
                {/* Model Architecture */}
                <div className="grid md:grid-cols-2 gap-8">
                  <div>
                    <h4 className="font-semibold text-lg text-white mb-4">
                      How It Works
                    </h4>
                    <ul className="space-y-3">
                      <li className="flex items-start gap-3 text-surface-300">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0 mt-2" />
                        <p>Uses same LSTM architecture as batter model, but with role-specific training</p>
                      </li>
                      <li className="flex items-start gap-3 text-surface-300">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0 mt-2" />
                        <p>Separate models for starters (≥80% starts) and relievers (&lt;80% starts)</p>
                      </li>
                      <li className="flex items-start gap-3 text-surface-300">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0 mt-2" />
                        <p>Uses 4-season sequences with IP-normalized statistics</p>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-surface-700/50 rounded-xl p-6">
                    <h4 className="font-semibold text-lg text-white mb-4">
                      Input Features
                    </h4>
                    <div className="grid grid-cols-2 gap-3 text-sm text-surface-300">
                      <div className="space-y-2">
                        <p className="font-medium">Core Metrics</p>
                        <ul className="space-y-1">
                          <li>Age</li>
                          <li>FIP</li>
                          <li>SIERA</li>
                          <li>ERA</li>
                          <li>IP</li>
                        </ul>
                      </div>
                      <div className="space-y-2">
                        <p className="font-medium">Rate Stats</p>
                        <ul className="space-y-1">
                          <li>Strikeout Rate</li>
                          <li>Walk Rate</li>
                          <li>Ground Ball Rate</li>
                          <li>Fly Ball Rate</li>
                        </ul>
                      </div>
                    </div>
                    <p className="text-xs text-slate-500 mt-4">All metrics normalized per inning pitched (except Age)</p>
                  </div>
                </div>

                {/* Sliding Window Example */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Sliding Window Prediction Process
                  </h4>
                  <div className="space-y-6">
                    <p className="text-sm text-surface-300">
                      Similar to the batter model, predictions use a sliding window of historical data,
                      but with 3-year sequences instead of 4.
                    </p>

                    <div className="space-y-4">
                      <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-emerald-500">
                        <h5 className="text-sm font-medium text-white mb-2">
                          2025 Prediction Uses (SP Example)
                        </h5>
                        <div className="grid grid-cols-5 gap-2 text-xs">
                          <div className="col-span-4 space-y-1 text-surface-300">
                            <p>2022: 2.44 ERA, 3.07 FIP (157 IP)</p>
                            <p>2023: 2.89 ERA, 3.21 FIP (168 IP)</p>
                            <p>2024: 2.63 ERA, 2.98 FIP (182 IP)</p>
                          </div>
                          <div className="text-emerald-600 dark:text-emerald-400">
                            → 2025: 2.75 ERA, 3.12 FIP (175 IP)
                          </div>
                        </div>
                      </div>

                      <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-blue-500">
                        <h5 className="text-sm font-medium text-white mb-2">
                          2026 Prediction Uses
                        </h5>
                        <div className="grid grid-cols-5 gap-2 text-xs">
                          <div className="col-span-4 space-y-1 text-surface-300">
                            <p>2023: 2.89 ERA, 3.21 FIP (168 IP)</p>
                            <p>2024: 2.63 ERA, 2.98 FIP (182 IP)</p>
                            <p className="text-emerald-600">2025: 2.75 ERA, 3.12 FIP (175 IP)</p>
                          </div>
                          <div className="text-emerald-600 dark:text-emerald-400">
                            → 2026: 2.82 ERA, 3.18 FIP (170 IP)
                          </div>
                        </div>
                      </div>

                      <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-purple-500">
                        <h5 className="text-sm font-medium text-white mb-2">
                          2027 Prediction Uses
                        </h5>
                        <div className="grid grid-cols-5 gap-2 text-xs">
                          <div className="col-span-4 space-y-1 text-surface-300">
                            <p>2024: 2.63 ERA, 2.98 FIP (182 IP)</p>
                            <p className="text-emerald-600">2025: 2.75 ERA, 3.12 FIP (175 IP)</p>
                            <p className="text-emerald-600">2026: 2.82 ERA, 3.18 FIP (170 IP)</p>
                          </div>
                          <div className="text-emerald-600 dark:text-emerald-400">
                            → 2027: 2.91 ERA, 3.24 FIP (165 IP)
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="text-sm text-surface-300 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                      <span className="font-medium">Note:</span> IP projections are dynamically adjusted based on 
                      predicted performance - better projected stats lead to more projected innings, reflecting 
                      real-world usage patterns.
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </CollapsibleSection>
        <CollapsibleSection
              title="Defensive Projections Models"
              subtitle="Position-specific defensive metrics"
              icon={FaBaseballBall}
            >
            <Card className="mt-12">
        <div className="flex justify-between items-center mb-8">
          <SubHeading icon={FaBaseballBall}>Model Overview</SubHeading>
          <a href="https://github.com/nielsjsc/LSTMLB/blob/main/models/defense.ipynb"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500 
                      hover:bg-emerald-400 text-white transition-colors text-sm">
            <FaGithub /> View Model Code
          </a>
        </div>

        <div className="space-y-8">
          {/* Position Group Models */}
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-surface-700/50 rounded-xl p-6">
              <h4 className="font-semibold text-lg text-white mb-4">
                Infield Model
              </h4>
              <div className="space-y-4">
                <p className="text-sm text-surface-300">
                  Covers 1B, 2B, 3B, SS positions
                </p>
                <div className="space-y-2">
                  <h5 className="text-sm font-medium text-white">Features</h5>
                  <ul className="text-sm space-y-1 text-surface-300">
                    <li>DRS/150</li>
                    <li>UZR/150</li>
                    <li>OAA/150</li>
                    <li>RngR/150</li>
                    <li>ErrR/150</li>
                    <li>DPR/150</li>
                    <li>Inn</li>
                    <li>Age</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-surface-700/50 rounded-xl p-6">
              <h4 className="font-semibold text-lg text-white mb-4">
                Outfield Model
              </h4>
              <div className="space-y-4">
                <p className="text-sm text-surface-300">
                  Covers LF, CF, RF positions
                </p>
                <div className="space-y-2">
                  <h5 className="text-sm font-medium text-white">Features</h5>
                  <ul className="text-sm space-y-1 text-surface-300">
                    <li>DRS/150</li>
                    <li>UZR/150</li>
                    <li>OAA/150</li>
                    <li>ARM/150</li>
                    <li>RngR/150</li>
                    <li>Inn</li>
                    <li>Age</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-surface-700/50 rounded-xl p-6">
              <h4 className="font-semibold text-lg text-white mb-4">
                Catcher Model
              </h4>
              <div className="space-y-4">
                <p className="text-sm text-surface-300">
                  Specialized catching metrics
                </p>
                <div className="space-y-2">
                  <h5 className="text-sm font-medium text-white">Features</h5>
                  <ul className="text-sm space-y-1 text-surface-300">
                    <li>DRS/150</li>
                    <li>FRM/150</li>
                    <li>rSB/150</li>
                    <li>rCERA/150</li>
                    <li>Inn</li>
                    <li>Age</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 my-8 border-l-4 border-blue-500">
            <h4 className="font-semibold text-lg text-white mb-2">
              Multi-Position Handling
            </h4>
            <p className="text-surface-300">
              Players who field multiple positions (e.g., utility infielders or multi-position outfielders) 
              are evaluated independently by each relevant position model. For example, a player who splits 
              time between 2B and SS will receive separate projections from the infield model for each 
              position. These position-specific projections are then weighted by playing time when 
              computing overall defensive value.
            </p>
          </div>
            </div>

      </Card>
        </CollapsibleSection>
        <CollapsibleSection
  title="Baserunning Projections Model"
  subtitle="Speed and baserunning value predictions"
  icon={FaRunning}
  defaultOpen={false}
>
  <div className="space-y-8">
    <Card>
      <div className="flex justify-between items-center mb-8">
        <SubHeading icon={FaRunning}>Model Overview</SubHeading>
        <div className="flex items-center gap-4">
          <a href="https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-brand-400 hover:text-emerald-400">
            Neural Networks Explained
          </a>
          <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-brand-400 hover:text-emerald-400">
            Understanding LSTMs
          </a>
          <a href="https://github.com/nielsjsc/LSTMLB/blob/main/models/baserunning.ipynb"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500 
                     hover:bg-emerald-400 text-white transition-colors text-sm">
            <FaGithub /> View Code
          </a>
        </div>
      </div>

      <div className="space-y-8">
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="font-semibold text-lg text-white mb-4">
              How It Works
            </h4>
            <ul className="space-y-3">
              <li className="flex items-start gap-3 text-surface-300">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0 mt-2" />
                <p>Uses 7-season sequences to capture long-term speed trends</p>
              </li>
              <li className="flex items-start gap-3 text-surface-300">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0 mt-2" />
                <p>Predicts both stolen base ability and general baserunning value</p>
              </li>
            </ul>
          </div>

          <div className="bg-surface-700/50 rounded-xl p-6">
            <h4 className="font-semibold text-lg text-white mb-4">
              Input Features
            </h4>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                  <p className="font-medium text-white">Value Metrics</p>
                  <ul className="space-y-1 text-sm text-surface-300">
                    <li>• Stolen Base Value (wSB)</li>
                    <li>• Ultimate Base Running (UBR)</li>
                    <li>• Double Play Value (wGDP)</li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <p className="font-medium text-white">Rate Stats</p>
                  <ul className="space-y-1 text-sm text-surface-300">
                    <li>• Stolen Base Rate</li>
                    <li>• Caught Stealing Rate</li>
                    <li>• Age</li>
                  </ul>
                </div>
              </div>
              <div className="pt-4 border-t border-slate-200 dark:border-slate-700">
                <div className="text-sm text-surface-300">
                  <span className="font-medium">Key Aspects:</span>
                  <ul className="mt-2 space-y-1">
                    <li>• All stats normalized per game played</li>
                    <li>• Requires 150 PA minimum for predictions</li>
                    <li>• Includes both stealing and non-stealing value</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="WAR Calculations"
          subtitle="Component-based WAR methodology"
          icon={FaCalculator}
          defaultOpen={false}
        >
          <div className="space-y-8">
            <Card>
              <div className="flex justify-between items-center mb-8">
                <SubHeading icon={FaCalculator}>WAR Calculation Components</SubHeading>
                <div className="flex items-center gap-4">
                  <a href="https://library.fangraphs.com/calculating-position-player-war-a-complete-example/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-brand-400 hover:text-emerald-400">
                    Position Player WAR Details
                  </a>
                  <a href="https://library.fangraphs.com/calculating-pitcher-war-a-complete-example/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-brand-400 hover:text-emerald-400">
                    Pitcher WAR Details
                  </a>
                </div>
              </div>

              <div className="space-y-8">
                {/* Position Player WAR */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Position Player WAR Components
                  </h4>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="col-span-2 bg-slate-100 dark:bg-slate-600/50 p-4 rounded-lg">
                      <div className="font-mono text-sm">
                        WAR = (Batting + Baserunning + Defense + Replacement) / RunsPerWin
                      </div>
                    </div>
                    <div className="space-y-3">
                      <h5 className="font-medium text-white">Offensive Value</h5>
                      <ul className="space-y-1 text-sm text-surface-300">
                        <li>• Batting: From batting model projections</li>
                        <li>• Baserunning: From baserunning model</li>
                      </ul>
                    </div>
                    <div className="space-y-3">
                      <h5 className="font-medium text-white">Defensive Value</h5>
                      <ul className="space-y-1 text-sm text-surface-300">
                        <li>• Fielding: From defensive model</li>
                        <li>• Position-specific adjustments</li>
                        <li>• Replacement level scaling</li>
                      </ul>
                    </div>
                  </div>
                </div>

                {/* Pitcher WAR */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Pitcher WAR Components
                  </h4>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="col-span-2 bg-slate-100 dark:bg-slate-600/50 p-4 rounded-lg">
                      <div className="font-mono text-sm">
                        WAR = (ReplacementFIP - ProjectedFIP) * InningsPitched / (9 * RunsPerWin)
                      </div>
                    </div>
                    <div className="space-y-3">
                      <h5 className="font-medium text-white">Starting Pitchers</h5>
                      <ul className="space-y-1 text-sm text-surface-300">
                        <li>• Base IP: 180 innings</li>
                        <li>• Range: 150-220 IP</li>
                        <li>• Scales with projected FIP</li>
                      </ul>
                    </div>
                    <div className="space-y-3">
                      <h5 className="font-medium text-white">Relief Pitchers</h5>
                      <ul className="space-y-1 text-sm text-surface-300">
                        <li>• Base IP: 65 innings</li>
                        <li>• Range: 50-80 IP</li>
                        <li>• Scales with projected FIP</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4">
                  <p className="text-sm text-surface-300">
                    <span className="font-medium">Note:</span> Our WAR calculations follow FanGraphs' 
                    methodology with adjustments for projections. Better projected performance leads 
                    to more playing time, creating a compound effect on WAR.
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="WAR to Dollar Value"
          subtitle="Tiered valuation system with future value adjustments"
          icon={FaDollarSign}
          defaultOpen={false}
        >
          <div className="space-y-8">
            <Card>
              <div className="flex justify-between items-center mb-8">
                <SubHeading icon={FaDollarSign}>Production Value Calculation</SubHeading>
              </div>

              <div className="space-y-8">
                {/* Value Tiers */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Tiered WAR Values
                  </h4>
                  <div className="grid md:grid-cols-3 gap-6">
                    <div className="space-y-2">
                      <h5 className="text-sm font-medium text-white">
                        Tier 1: 0-2 WAR
                      </h5>
                      <div className="bg-white dark:bg-slate-800 p-3 rounded text-sm">
                        <p className="text-surface-300">$8M per WAR</p>
                        <p className="text-xs text-slate-500 mt-1">Bench to Average Starter</p>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <h5 className="text-sm font-medium text-white">
                        Tier 2: 2-4 WAR
                      </h5>
                      <div className="bg-white dark:bg-slate-800 p-3 rounded text-sm">
                        <p className="text-surface-300">$9M per WAR</p>
                        <p className="text-xs text-slate-500 mt-1">Above Average to All-Star</p>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h5 className="text-sm font-medium text-white">
                        Tier 3: 4+ WAR
                      </h5>
                      <div className="bg-white dark:bg-slate-800 p-3 rounded text-sm">
                        <p className="text-surface-300">$10M per WAR</p>
                        <p className="text-xs text-slate-500 mt-1">All-Star to MVP Level</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Example Calculation */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Example: 6 WAR Player Value (2025)
                  </h4>
                  <div className="space-y-3 text-sm text-surface-300">
                    <div className="grid grid-cols-2 gap-2">
                      <div>First 2 WAR × $8M</div>
                      <div className="text-emerald-600">$16M</div>
                      <div>Next 2 WAR × $9M</div>
                      <div className="text-emerald-600">$18M</div>
                      <div>Final 2 WAR × $10M</div>
                      <div className="text-emerald-600">$20M</div>
                      <div className="font-medium">Total Value</div>
                      <div className="font-medium text-emerald-600">$54M</div>
                    </div>
                  </div>
                </div>

                {/* Future Value */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Future Value Adjustment
                  </h4>
                  <div className="space-y-2 text-sm text-surface-300">
                    <p>Base Year: 2025</p>
                    <p>Annual Inflation: 4%</p>
                    <div className="bg-white dark:bg-slate-800 p-3 rounded mt-3">
                      <p className="font-mono">FutureValue = BaseValue × (1.04)^YearsFromBase</p>
                    </div>
                  </div>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4">
                  <p className="text-sm text-surface-300">
                    This tiered system reflects how MLB teams value roster spots, with premium 
                    values for elite talent that can concentrate production in fewer positions.
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="Arbitration Calculations"
          subtitle="Service time-based salary progression"
          icon={FaBalanceScale}
          defaultOpen={false}
        >
          <div className="space-y-8">
            <Card>
              <div className="flex justify-between items-center mb-8">
                <SubHeading icon={FaBalanceScale}>Arbitration Salary Model</SubHeading>
              </div>

              <div className="space-y-8">
                {/* Service Time Tiers */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Service Time Tiers
                  </h4>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div>
                        <h5 className="text-sm font-medium text-white">
                          Pre-Arbitration (0-3 years)
                        </h5>
                        <div className="bg-white dark:bg-slate-800 p-3 rounded mt-2 text-sm">
                          <p className="text-surface-300">Minimum: $720K</p>
                          <p className="text-xs text-slate-500 mt-1">Cannot decrease from previous year</p>
                        </div>
                      </div>
                      <div>
                        <h5 className="text-sm font-medium text-white">
                          Arbitration 1 (3-4 years)
                        </h5>
                        <div className="bg-white dark:bg-slate-800 p-3 rounded mt-2 text-sm">
                          <p className="text-surface-300">25% of market value</p>
                          <p className="text-xs text-slate-500 mt-1">Minimum: $1M (Regular), $1.2M (Super Two)</p>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div>
                        <h5 className="text-sm font-medium text-white">
                          Arbitration 2 (4-5 years)
                        </h5>
                        <div className="bg-white dark:bg-slate-800 p-3 rounded mt-2 text-sm">
                          <p className="text-surface-300">33% of market value</p>
                          <p className="text-xs text-slate-500 mt-1">Minimum: $2.5M</p>
                        </div>
                      </div>
                      <div>
                        <h5 className="text-sm font-medium text-white">
                          Arbitration 3 (5-6 years)
                        </h5>
                        <div className="bg-white dark:bg-slate-800 p-3 rounded mt-2 text-sm">
                          <p className="text-surface-300">50% of market value</p>
                          <p className="text-xs text-slate-500 mt-1">Minimum: $4M</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Calculation Example */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Calculation Example
                  </h4>
                  <div className="space-y-3">
                    <p className="text-sm text-surface-300">
                      Player in Arbitration 2 with $10M market value:
                    </p>
                    <div className="bg-white dark:bg-slate-800 p-4 rounded font-mono text-sm">
                      <div className="grid grid-cols-2 gap-2 text-surface-300">
                        <div>Market Value</div>
                        <div>$10,000,000</div>
                        <div>Arb-2 Rate (33%)</div>
                        <div>× 0.33</div>
                        <div className="border-t pt-2">Raw Arb Value</div>
                        <div className="border-t pt-2">$3,300,000</div>
                      </div>
                      <div className="mt-4 text-emerald-600 dark:text-emerald-400">
                        Final Value = max($3.3M, $2.5M minimum, previous × 1.25)
                      </div>
                    </div>
                  </div>
                </div>

                {/* Key Rules */}
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4">
                  <h4 className="font-medium text-white mb-2">Key Rules</h4>
                  <ul className="space-y-1 text-sm text-surface-300">
                    <li>• Cannot decrease unless performance significantly declines</li>
                    <li>• Guaranteed 25% raise if performance maintained</li>
                    <li>• Super Two players get extra arbitration year</li>
                    <li>• All values adjusted for future year inflation</li>
                  </ul>
                </div>
              </div>
            </Card>
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="Trade Value Calculator"
          subtitle="Long-term surplus value analysis"
          icon={FaExchangeAlt}
          defaultOpen={false}
        >
          <div className="space-y-8">
            <Card>
              <div className="flex justify-between items-center mb-8">
                <SubHeading icon={FaExchangeAlt}>Trade Value Methodology</SubHeading>
              </div>

              <div className="space-y-8">
                {/* Core Formula */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Core Calculation
                  </h4>
                  <div className="font-mono text-sm space-y-2">
                    <div className="bg-white dark:bg-slate-800 p-4 rounded">
                      <p className="text-emerald-600">Trade Value = Σ (Production Value - Contract Value)</p>
                      <p className="text-xs text-slate-500 mt-2">Summed across all team-control years</p>
                    </div>
                  </div>
                </div>

                {/* Example Calculation */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Example: 3 WAR Player with 2 Years of Control
                  </h4>
                  <div className="space-y-6">
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <h5 className="text-sm font-medium text-white mb-3">Year 1</h5>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between text-surface-300">
                            <span>Production Value:</span>
                            <span>$25.0M</span>
                          </div>
                          <div className="flex justify-between text-surface-300">
                            <span>Contract Value:</span>
                            <span>$12.5M</span>
                          </div>
                          <div className="flex justify-between text-emerald-600 pt-2 border-t">
                            <span>Surplus Value:</span>
                            <span>$12.5M</span>
                          </div>
                        </div>
                      </div>
                      <div>
                        <h5 className="text-sm font-medium text-white mb-3">Year 2</h5>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between text-surface-300">
                            <span>Production Value:</span>
                            <span>$26.0M</span>
                          </div>
                          <div className="flex justify-between text-surface-300">
                            <span>Contract Value:</span>
                            <span>$15.0M</span>
                          </div>
                          <div className="flex justify-between text-emerald-600 pt-2 border-t">
                            <span>Surplus Value:</span>
                            <span>$11.0M</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg">
                      <div className="flex justify-between text-lg font-medium text-white">
                        <span>Total Trade Value:</span>
                        <span>$23.5M</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Value Factors */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Key Value Drivers
                  </h4>
                  <div className="grid md:grid-cols-3 gap-6 text-sm">
                    <div className="space-y-2">
                      <h5 className="font-medium text-white">Years of Control</h5>
                      <p className="text-surface-300">
                        More team control years means more potential surplus value
                      </p>
                    </div>
                    <div className="space-y-2">
                      <h5 className="font-medium text-white">Contract Status</h5>
                      <p className="text-surface-300">
                        Pre-arb and early arb years typically generate highest surplus
                      </p>
                    </div>
                    <div className="space-y-2">
                      <h5 className="font-medium text-white">Performance Level</h5>
                      <p className="text-surface-300">
                        Elite players can generate surplus even with large contracts
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4">
                  <p className="text-sm text-surface-300">
                    This system quantifies how teams evaluate trade targets, combining our WAR and 
                    contract projections to estimate a player's total trade value. Young, controllable
                    talent typically carries the highest trade value due to potential surplus value
                    during pre-arbitration and arbitration years.
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </CollapsibleSection>
        <CollapsibleSection
          title="Prospect Trade Values"
          subtitle="FV and consensus ranking-based valuation"
          icon={FaStar}
          defaultOpen={false}
        >
          <div className="space-y-8">
            <Card>
              <div className="flex justify-between items-center mb-8">
                <SubHeading icon={FaStar}>Prospect Value System</SubHeading>
              </div>

              <div className="space-y-8">
                {/* Rankings Source */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Consensus Rankings
                  </h4>
                  <div className="space-y-3">
                    <p className="text-sm text-surface-300">
                      We use a comprehensive composite prospect ranking system aggregated from multiple 
                      major scouting outlets and publications, thanks to the work of{' '} 
                      <a 
                        href="https://www.reddit.com/r/fantasybaseball/comments/1ibccdo/composite_prospect_list_2025/"
                        className="text-brand-400 hover:text-emerald-400"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        u/Phrim 
                      </a>
                       , on Reddit.
                    </p>
                  </div>
                </div>

                {/* FV Grade Tiers */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Future Value Base Values
                  </h4>
                  <div className="grid md:grid-cols-4 gap-4">
                    <div className="bg-white dark:bg-slate-800 p-3 rounded">
                      <div className="text-lg font-medium text-emerald-600">70 FV</div>
                      <div className="text-sm text-surface-300">$180M</div>
                      <div className="text-xs text-slate-500 mt-1">Elite Prospects</div>
                    </div>
                    <div className="bg-white dark:bg-slate-800 p-3 rounded">
                      <div className="text-lg font-medium text-emerald-600">60-65 FV</div>
                      <div className="text-sm text-surface-300">$90M-$120M</div>
                      <div className="text-xs text-slate-500 mt-1">Top Prospects</div>
                    </div>
                    <div className="bg-white dark:bg-slate-800 p-3 rounded">
                      <div className="text-lg font-medium text-emerald-600">50-55 FV</div>
                      <div className="text-sm text-surface-300">$28M-$55M</div>
                      <div className="text-xs text-slate-500 mt-1">Average to Above</div>
                    </div>
                    <div className="bg-white dark:bg-slate-800 p-3 rounded">
                      <div className="text-lg font-medium text-emerald-600">40-45 FV</div>
                      <div className="text-sm text-surface-300">$4M-$12M</div>
                      <div className="text-xs text-slate-500 mt-1">Depth Prospects</div>
                    </div>
                  </div>
                </div>

                {/* Ranking Adjustments */}
                <div className="bg-surface-700/50 rounded-xl p-6">
                  <h4 className="font-semibold text-lg text-white mb-4">
                    Ranking Multipliers
                  </h4>
                  <div className="grid md:grid-cols-3 gap-6">
                    <div className="space-y-2">
                      <h5 className="text-sm font-medium text-white">Top 100</h5>
                      <div className="text-sm text-surface-300">
                        0.9 → 0.5 multiplier
                        <p className="text-xs text-slate-500 mt-1">Linear decrease from #1 to #100</p>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <h5 className="text-sm font-medium text-white">Extended (101-500)</h5>
                      <div className="text-sm text-surface-300">
                        0.5 → 0.3 multiplier
                        <p className="text-xs text-slate-500 mt-1">Linear decrease from #101 to #500</p>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <h5 className="text-sm font-medium text-white">Unranked</h5>
                      <div className="text-sm text-surface-300">
                        0.3 multiplier
                        <p className="text-xs text-slate-500 mt-1">Minimum value floor</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Recent Graduate Note */}
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4">
                  <div className="space-y-2">
                    <h4 className="font-medium text-white">Recent MLB Graduates</h4>
                    <p className="text-sm text-surface-300">
                      For prospects who recently reached MLB, we gradually transition from their prospect 
                      value to their projected MLB value over their first 300 games for position players, 45 for SP, and 65 for RP, providing smoother 
                      value progression during their early career.
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </CollapsibleSection>

</div>
    </section>
    
  )
}

export default TechnicalOverview
