Overview:

This project aims to develop machine learning models to predict the future value of MLB/MiLB players in terms of Wins Above Replacement (WAR). We will then combine these predictions with real salary data to produce a player value statistic.

Features:
- Data Collection:
  - Comprehensive scraping of MLB player statistics and salary data.
  - Inclusion of minor league performance data and prospect rankings.

- Data Processing: 
  - Cleaning and merging of datasets to ensure high-quality input for modeling.
  - Feature engineering to capture relevant player metrics over time.
  - Handling of unique cases such as players with the same name and tracking player debuts accurately.

- Model Development:
  - Implementation of machine learning models (LSTM, Linear Regression) for predicting future WAR values.
  - Separate modeling for hitters and pitchers to improve accuracy.

- Evaluation:
  - Use of metrics like Mean Squared Error (MSE) and R² for model performance assessment.


Prerequisites:

- Python 3.7+
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `torch`, `beautifulsoup4`, `selenium`

Installation
1. Clone the repository:
   git clone https://github.com/Nielsjsc/MLBTradeSim.git
   cd MLBTradeSim
2. Install the required packages:
   bash
   pip install -r requirements.txt
   

Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes.

This project is licensed under the MIT License
