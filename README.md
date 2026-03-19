# Short-Term Stock Price Predictor

A machine learning pipeline built with Python to predict the short-term future closing prices of stocks using historical stock data from Yahoo Finance.

## Features
- **Dynamic Data Scraping:** Automatically fetches the most recent historical market data using `yfinance`.
- **Advanced Technical Indicators:** Automatically engineers features like 10-day and 50-day Moving Averages, Volatility, Momentum, and vital Price Ratios (Close-to-Open and High-to-Low).
- **Multiple Models:** Leverages classical **Linear Regression** alongside complex ensemble methods like **Random Forest Regressor** to evaluate benchmark predictions against robust model forecasts.
- **Robust Error Handling:** Implements automatic exponential backoff style retries natively for consistent Yahoo Finance API connections.
- **Evaluation & Visualization:** Evaluates predictions utilizing key metrics (MSE, RMSE, MAE, R²) and graphs the testing set prices directly in matplotlib for easy visual comparison.

## Requirements
Ensure you have the following packages installed:
```bash
pip install yfinance pandas numpy matplotlib scikit-learn
```

## Usage
The script uses command-line arguments so you can analyze any stock over any specific time range dynamically.

### Basic Usage (Defaults to `AAPL` over the past 4 years):
```bash
python Future_stock.py
```

### Advanced Usage with Arguments:
You can flag `--stock`, `--start`, and `--end` dynamic overrides.

```bash
python Future_stock.py --stock MSFT --start 2021-01-01 --end 2023-12-31
```

* **`--stock`**: The stock ticker symbol to look up (e.g. `TSLA`, `GOOG`, `NVDA`).
* **`--start`**: The start bound for fetching historical data (formatted `YYYY-MM-DD`). 
* **`--end`**: The end bound for fetching historical data (formatted `YYYY-MM-DD`).

## Output Example
The script logs detailed run progression into standard output and saves a high-resolution PNG image chart comparing Actual Prices against Linear Regression and Random Forest Predicted Prices for your testing dataset limit. It will also print the predicted stock close value for the next actionable trading day at the end of the log output.
