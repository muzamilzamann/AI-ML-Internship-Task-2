import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments():
    """
    Parses command line arguments to allow dynamic stock selection and date ranges.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Advanced Short-Term Stock Price Predictor")
    parser.add_argument("--stock", type=str, default="AAPL", help="Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)")
    
    # Default to 4 years ago for the start date
    default_start = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
    default_end = datetime.now().strftime('%Y-%m-%d')
    
    parser.add_argument("--start", type=str, default=default_start, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=default_end, help="End date (YYYY-MM-DD)")
    
    return parser.parse_args()


import time

def download_data(stock: str, start: str, end: str) -> pd.DataFrame:
    """
    Downloads historical stock data from Yahoo Finance with retries.
    
    Args:
        stock (str): Ticker symbol.
        start (str): Start date in YYYY-MM-DD format.
        end (str): End date in YYYY-MM-DD format.
        
    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """
    logging.info(f"Downloading historical data for {stock} from {start} to {end}...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = yf.download(stock, start=start, end=end, auto_adjust=False)
            if not data.empty:
                logging.info(f"Successfully downloaded {len(data)} rows of data.")
                return data
            else:
                logging.warning(f"Attempt {attempt + 1}: Received empty data for {stock}. Retrying...")
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}: Failed to download data due to error: {e}")
            
        time.sleep(2)
        
    raise ValueError(f"No data downloaded for {stock} after {max_retries} attempts. Please check the ticker symbol or date range.")



def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering to the raw stock data to compute technical indicators
    like Moving Averages, Volatility, Momentum, and Price Ratios.
    
    Args:
        data (pd.DataFrame): Raw historical stock data.
        
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    logging.info("Engineering features: Moving Averages, Volatility, Momentum, etc...")
    df = data.copy()
    
    # 1. Target variable: Next day's Close price
    df["Next_Close"] = df["Close"].shift(-1)
    
    # 2. Moving Averages
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    
    # 3. Daily Returns and Volatility
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Daily_Return"].rolling(window=10).std()
    
    # 4. Momentum (Close price relative to a past close price)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    
    # 5. Price to Open/High/Low ratios
    df["Close_to_Open"] = df["Close"] / df["Open"]
    df["High_to_Low"] = df["High"] / df["Low"]
    
    # Drop rows with NaN values created by moving averages and shifting
    initial_len = len(df)
    df = df.dropna()
    logging.info(f"Dropped {initial_len - len(df)} rows due to NaN values from feature calculations.")
    
    return df


def train_and_evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series, y_test: pd.Series, 
                             model_name: str = "Linear Regression"):
    """
    Trains a predictive model and evaluates its performance on the test set.
    
    Args:
        X_train, X_test: Feature datasets for training and testing.
        y_train, y_test: Target variables for training and testing.
        model_name (str): The name of the model to use.
        
    Returns:
        model, predictions: The trained model and test set predictions.
    """
    logging.info(f"Training {model_name}...")
    
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        # Using 100 trees, which is standard, and configuring random_state for reproducibility
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
        
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on test set
    predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    logging.info(f"--- {model_name} Evaluation ---")
    logging.info(f"  MSE:  {mse:.4f}")
    logging.info(f"  RMSE: {rmse:.4f}")
    logging.info(f"  MAE:  {mae:.4f}")
    logging.info(f"  R2:   {r2:.4f}")
    
    return model, predictions


def plot_results(y_test: pd.Series, lr_predictions: np.ndarray, 
                 rf_predictions: np.ndarray, stock: str):
    """
    Plots the actual vs predicted stock prices for multiple models.
    
    Args:
        y_test: Actual target values.
        lr_predictions: Predictions from Linear Regression.
        rf_predictions: Predictions from Random Forest.
        stock: The stock ticker symbol.
    """
    logging.info("Generating comparison plots...")
    plt.figure(figsize=(14, 7))
    
    # Use index of y_test for x-axis if it's datetime, otherwise just simple sequence
    x_axis = np.arange(len(y_test))
    
    plt.plot(x_axis, y_test.values, label="Actual Price", color='black', linewidth=2)
    plt.plot(x_axis, lr_predictions, label="Linear Regression Predicted", color='blue', alpha=0.7)
    plt.plot(x_axis, rf_predictions, label="Random Forest Predicted", color='red', alpha=0.7)
    
    plt.title(f"{stock} Actual vs Predicted Closing Prices")
    plt.xlabel("Testing Days")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_filename = f"{stock}_prediction_results.png"
    plt.savefig(output_filename, dpi=300)
    logging.info(f"Plot saved successfully as '{output_filename}'")
    plt.close()


def predict_future(model, latest_features: pd.Series, model_name: str) -> float:
    """
    Predicts the next day's closing price using the latest available features.
    
    Args:
        model: Trained machine learning model.
        latest_features (pd.Series): The most recent row of features.
        model_name (str): Name of the model (for logging).
        
    Returns:
        float: The predicted next day closing price.
    """
    # reshape(1, -1) because we're predicting a single sample
    prediction = model.predict(latest_features.values.reshape(1, -1))
    logging.info(f"{model_name} - Predicted Next Day Closing Price: ${prediction[0]:.2f}")
    return prediction[0]


def main():
    """
    Main function executing the pipeline: Data Download -> Feature Engineering -> 
    Modeling -> Evaluation -> Forecasting -> Visualization.
    """
    args = parse_arguments()
    
    try:
        # 1. Download Data
        data = download_data(args.stock, args.start, args.end)
        
        # 2. Feature Engineering
        df = feature_engineering(data)
        
        # 3. Prepare Train and Test splits
        features = [
            "Open", "High", "Low", "Volume", 
            "MA_10", "MA_50", "Daily_Return", 
            "Volatility_10", "Momentum_10", 
            "Close_to_Open", "High_to_Low"
        ]
        
        X = df[features]
        y = df["Next_Close"]
        
        # We use shuffle=False because stock data is a time series
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        logging.info(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
        logging.info(f"Testing set shape:  X={X_test.shape}, y={y_test.shape}")
        
        # (Optional) Data Scaling is sometimes necessary for algorithms like SVR or Neural Networks.
        # It doesn't hurt Linear Regression, but Random Forests don't necessarily need it.
        # We'll skip standard scaling here to maintain interpretation simplicity.
        
        # 4. Train Models
        lr_model, lr_preds = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, model_name="Linear Regression"
        )
        
        rf_model, rf_preds = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, model_name="Random Forest"
        )
        
        # 5. Plotting
        plot_results(y_test, lr_preds, rf_preds, args.stock)
        
        # 6. Predict next day
        logging.info("-" * 50)
        logging.info("FORECASTING NEXT DAY'S CLOSE PRICE")
        logging.info("-" * 50)
        
        # The very last row of the feature set
        latest_features = X.iloc[-1]
        logging.info(f"Features corresponding to the latest date:\n{latest_features.to_string()}")
        
        predict_future(lr_model, latest_features, "Linear Regression")
        predict_future(rf_model, latest_features, "Random Forest")
        
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()