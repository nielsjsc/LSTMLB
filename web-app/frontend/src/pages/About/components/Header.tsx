import { FaGithub, FaChartLine, FaBrain, FaStar, FaExchangeAlt, FaCode, FaExclamationTriangle, FaCalculator, FaLightbulb } from 'react-icons/fa'


const Header = () => {
    return (
      <section id="header" className="flex flex-col justify-center py-12">
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h1 className="font-display text-5xl font-extrabold mb-6 bg-clip-text text-transparent bg-gradient-brand pb-3 tracking-tight">
              LONGBALL
            </h1>
            <p className="text-2xl text-gray-300 font-light max-w-3xl mx-auto">
              Open Source Baseball Analytics Platform
            </p>
            <p className="text-lg text-gray-400 mt-4 max-w-2xl mx-auto">
              MLB projections, prospect valuations, and trade analysis - 
              free and open source for the baseball community.
            </p>
          </div>
  
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            <div className="bg-gray-50 rounded-lg p-5 border border-gray-200 hover:border-gray-300 transition-all">
              <div className="text-brand-500 mb-4">
                <FaChartLine className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">MLB Projections</h3>
              <p className="text-gray-500">
                Long-term career projections using specialized models for hitting, pitching, and fielding
              </p>
            </div>
  
            <div className="bg-gray-50 rounded-lg p-5 border border-gray-200 hover:border-gray-300 transition-all">
              <div className="text-brand-500 mb-4">
                <FaExchangeAlt className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Trade Analysis</h3>
              <p className="text-gray-500">
                Comprehensive trade simulator with WAR projections and surplus value calculations
              </p>
            </div>

            <div className="bg-gray-50 rounded-lg p-5 border border-gray-200 hover:border-gray-300 transition-all">
              <div className="text-brand-500 mb-4">
                <FaStar className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Prospect Values</h3>
              <p className="text-gray-500">
                FV and ranking-based prospect valuations with dynamic graduate adjustments
              </p>
            </div>
  
            <div className="bg-gray-50 rounded-lg p-5 border border-gray-200 hover:border-gray-300 transition-all">
              <div className="text-brand-500 mb-4">
                <FaCode className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Open Source</h3>
              <p className="text-gray-500">
                All models and code freely available for community use and contribution
              </p>
            </div>
          </div>
  
          <div className="flex justify-center">
            <a 
              href="https://github.com/nielsjsc/LSTMLB" 
              target="_blank" 
              rel="noopener noreferrer"
              className="group px-8 py-3 rounded-lg bg-brand-500 hover:bg-brand-400 text-white font-semibold transition-colors flex items-center gap-2"
            >
              <FaGithub className="text-xl" />
              <span>View on GitHub</span>
            </a>
          </div>
        </div>

        {/* Philosophy Section */}
        <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 mt-16">
          <div className="space-y-8">
            {/* Introduction */}
            <div className="border-l-2 border-brand-400 bg-white rounded-lg p-6 border border-gray-200">
              <h2 className="text-4xl font-bold mb-8 bg-clip-text text-transparent bg-gradient-brand">
                The Story Behind LongBall Analytics
              </h2>
              <div className="prose prose-lg prose-invert max-w-none space-y-6">
                <p className="text-xl text-gray-300 leading-relaxed">
                  This project started as an attempt to create a trade value metric for players, as this is not a publicly available statistic yet it's one that I'm always curious about. What do we need in order to calculate trade value? Projected future WAR and salary data, basically how much on field value are you expected to produce, and how much money are you owed. There are existing robust mlb projection models, with their results for the subsequent year and sometimes 3 years available on fangraphs. However, if we want to know how much Juan Soto is worth, we need to know how much value he will be projected to produce in 2035, not just 2026.
                </p>
              </div>
            </div>

            {/* Model Creation */}
            <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
              <div className="flex items-start gap-4">
                <FaBrain className="h-8 w-8 text-blue-400 mt-1 flex-shrink-0" />
                <div className="prose prose-lg prose-invert max-w-none">
                  <p className="text-gray-300 leading-relaxed">
                    This led me to decide to create my own projection models. I have experience with deep learning, and while unorthodox in the baseball projection world (as far as I can tell) I wanted to give it a try given the current AI boom in other fields. I landed on using LSTM's, as baseball statistics are naturally sequential data. In the sections below I explain the details of the models I trained, but with these models I produced projections for the next 15 years for every single MLB player.
                  </p>
                </div>
              </div>
            </div>

            {/* Warning Section */}
            <div className="bg-red-500/10 rounded-lg p-6 border border-red-500/20">
              <div className="flex items-start gap-4">
                <FaExclamationTriangle className="h-8 w-8 text-red-400 mt-1 flex-shrink-0" />
                <div>
                  <h3 className="text-2xl font-bold text-red-400 mb-4">Casual Use of Projections/Values</h3>
                  <div className="prose prose-lg prose-invert max-w-none">
                    <p className="text-gray-300 leading-relaxed">
                      I want to make it very clear I am NOT confident in my models abilities to predict player performance deep into the future, and if you look closely at some of the long term projections, I'm sure you will be able to find many things you disagree with. Do not look at my data and say "Hey I didn't know Jackson Chourio is going to put up 7.1 WAR in 2029" because that likely won't happen, it's just a model attempting to minimize the loss of its guesses.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Disclaimers */}
            <div className="bg-orange-500/10 rounded-lg p-6 border border-orange-500/20">
              <div className="flex items-start gap-4">
                <FaExclamationTriangle className="h-8 w-8 text-orange-400 mt-1 flex-shrink-0" />
                <div>
                  <h3 className="text-2xl font-bold text-orange-400 mb-4">Disclaimers</h3>
                  <div className="prose prose-lg prose-invert max-w-none space-y-6">
                    <p className="text-gray-300 leading-relaxed">
                      I personally believe the hitting projections to be fairly accurate, however the pitching projections are not great. I hypothesize this is partially due to the much smaller sample size of pitching data available, especially with starting pitchers. Specifically, I found it almost always projects a performance decrease for pitchers in each subsequent year, no matter their age, and sometimes this decline is quite steep. For this reason, our pitchers are (in my opinion) undervalued in their trade values and the trade simulator.
                    </p>
                    <p className="text-gray-300 leading-relaxed">
                      Additionally, the models work poorly in predicting players with small sample sizes. However I would rather not leave them out of the projections. I have experimented with implementing more complex models that combine minor league statistics to predict performance, but they were disappointing.
                    </p>
                    <p className="text-gray-300 leading-relaxed">
                      For these reasons, take all of my projections and trade simulations with a very large, very flaky grain of salt.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Calculation Methodology */}
            <div className="bg-blue-500/10 rounded-lg p-6 border border-blue-500/20">
              <div className="flex items-start gap-4">
                <FaCalculator className="h-8 w-8 text-blue-400 mt-1 flex-shrink-0" />
                <div>
                  <h3 className="text-2xl font-bold text-blue-400 mb-4">Value Calculations</h3>
                  <div className="prose prose-lg prose-invert max-w-none space-y-8">
                    <p className="text-gray-600">
                      Regarding the calculation of values, I took quite a few liberties with my projections that I feel everyone should be aware of.
                    </p>
                    
                    <div>
                      <h4 className="text-xl font-semibold text-blue-400 mb-2">Normalization of games played</h4>
                      <p className="text-gray-300 leading-relaxed">
                        Hitters play 150 games (catchers 135), starters pitch 32 games, relievers pitch 65 innings. While I understand injuries/playing time are an integral part of trade value, this is difficult data to obtain and at least with injuries, very difficult to predict. This means all of my models were trained on rate statistics with weights for games played in previous seasons.
                      </p>
                    </div>

                    <div>
                      <h4 className="text-xl font-semibold text-blue-400 mb-2">Change in WAR to Dollar calculation</h4>
                      <p className="text-gray-300 leading-relaxed">
                        Typically a dollar value assigned to a player is the current market rate for 1 WAR multiplied by their WAR. However, I don't believe the relationship of WAR to dollars should be linear. Two 2 WAR players aren't worth a 4 WAR player. A 4 WAR player produces the value of those two while providing an extra roster spot. For this reason I gave slight weights to more valuable players, as explained in the WAR to Dollar tab.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Have Fun Section */}
            <div className="bg-brand-50 rounded-lg p-6 border border-brand-200">
              <div className="flex items-start gap-4">
                <FaLightbulb className="h-8 w-8 text-brand-500 mt-1 flex-shrink-0" />
                <div>
                  <h3 className="text-2xl font-bold text-brand-500 mb-4">Have Fun!</h3>
                  <div className="prose prose-lg prose-invert max-w-none space-y-4">
                    <p className="text-gray-300 leading-relaxed">
                      I hope you all are able to use this as a fun tool to get a glimpse into how my models project your favorite players will perform in the near/far future!
                    </p>
                    <p className="text-gray-300 leading-relaxed">
                      Additionally, all of my code from this project is publicly available on my github, and you are free to clone my repo and experiment with training the models yourself, they aren't super computationally expensive so go crazy, I would love to see some people try to make some improvements to the models!
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      </section>
    )
}

export default Header