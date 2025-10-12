# === Stock Chart + Deep Learning Prediction Tool v0.5 ===

import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


# =======================
# DATA FETCHING
# =======================

def get_stock_data(ticker):
    """Download 2 years of stock data and clean it up"""
    print(f"Getting {ticker} data...")
    data = yf.download(ticker, period="2y", progress=False)

    if data.empty:
        print(f"No data found for {ticker}")
        return None

    if data.columns.nlevels > 1:
        data.columns = data.columns.droplevel(1)

    for column in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna()

    print(f"Got {len(data)} days of data")
    return data


def group_days_simple(data, num_days):
    """Group daily data into combined candles"""
    if num_days == 1:
        return data

    grouped_data = []
    dates = []

    for i in range(0, len(data), num_days):
        chunk = data.iloc[i:i+num_days]
        if len(chunk) > 0:
            combined = {
                'Open': chunk['Open'].iloc[0],
                'High': chunk['High'].max(),
                'Low': chunk['Low'].min(),
                'Close': chunk['Close'].iloc[-1],
                'Volume': chunk['Volume'].sum()
            }
            grouped_data.append(combined)
            dates.append(chunk.index[-1])

    return pd.DataFrame(grouped_data, index=dates)


def make_candlestick_chart(data, ticker, days_per_candle=1):
    """Show candlestick chart"""
    if data is None:
        print("No data to show!")
        return

    if days_per_candle > 1:
        print(f"Creating {days_per_candle}-day candles...")
        data = group_days_simple(data, days_per_candle)

    recent_data = data.tail(100)

    period_name = "Daily"
    if days_per_candle == 5:
        period_name = "Weekly"
    elif days_per_candle == 20:
        period_name = "Monthly"

    print(f"Making {period_name.lower()} chart...")
    mpf.plot(
        recent_data,
        type='candle',
        title=f'{ticker} Stock Price ({period_name})',
        ylabel='Price ($)',
        volume=True,
        style='charles',
        figsize=(12, 6)
    )


def make_boxplot_chart(data, ticker, window_size=5):
    """Make a boxplot using moving windows"""
    if data is None:
        print("No data to show!")
        return

    if len(data) < window_size:
        print(f"Not enough data for {window_size}-day window!")
        return

    print(f"Creating boxplot with {window_size}-day moving windows...")

    windows, window_labels = [], []

    for i in range(len(data) - window_size + 1):
        window_data = data.iloc[i:i+window_size]
        windows.append(window_data['Close'].tolist())
        window_labels.append(str(window_data.index[-1].date()))

    if len(windows) > 50:
        step = len(windows) // 50
        windows = windows[::step]
        window_labels = window_labels[::step]

    plt.figure(figsize=(14, 7))
    box_plot = plt.boxplot(windows, showmeans=True, patch_artist=True)
    for patch in box_plot['boxes']:
        patch.set_facecolor('lightblue')

    plt.title(f"{ticker} Stock Prices - {window_size}-Day Moving Window Boxplot")
    plt.xlabel(f"Trading Periods ({window_size} days each)")
    plt.ylabel("Closing Price ($)")
    tick_positions = range(1, len(window_labels) + 1, max(1, len(window_labels) // 10))
    tick_labels = [window_labels[i-1] for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =======================
# ARIMA MODEL
# =======================

def train_arima_model(prices, future_days=5, order=(1, 1, 1)):
    """Train ARIMA model - simpler than SARIMA"""
    print(f"  Training ARIMA - order{order}...")
    try:
        model = ARIMA(prices, order=order)
        fitted_model = model.fit()
        forecast = fitted_model.get_forecast(steps=future_days)
        predictions = forecast.predicted_mean.values
        print(f"  ARIMA trained")
        return predictions
    except Exception as e:
        print(f"  ARIMA failed: {e}")
        return None


# =======================
# SARIMA MODELS
# =======================

def train_sarima_model(prices, future_days=5, order=(2, 1, 1), seasonal_order=(1, 1, 1, 5)):
    """Train SARIMA model"""
    print(f"  Training SARIMA - order{order}, seasonal{seasonal_order}...")
    try:
        model = SARIMAX(prices, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        forecast = fitted_model.get_forecast(steps=future_days)
        predictions = forecast.predicted_mean.values
        print(f"  SARIMA trained")
        return predictions
    except Exception as e:
        print(f"  SARIMA failed: {e}")
        return None


# =======================
# LSTM MODELS
# =======================

def prepare_lstm_data(data, look_back_days=60, future_days=5):
    """Prepare data for LSTM"""
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(look_back_days, len(scaled_prices) - future_days + 1):
        X.append(scaled_prices[i-look_back_days:i, 0])
        y.append(scaled_prices[i:i+future_days, 0])

    X = np.array(X).reshape(-1, look_back_days, 1)
    y = np.array(y)
    return X, y, scaler


def build_lstm_model(look_back_days=60, future_days=5, lstm_units=50, dropout=0.2):
    """Build LSTM model"""
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(look_back_days, 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout))
    model.add(Dense(future_days))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm_model(data, future_days=5, look_back_days=60, lstm_units=50, dropout=0.2, epochs=10):
    """Train LSTM model"""
    print(f"  Training LSTM - units={lstm_units}, dropout={dropout}, epochs={epochs}...")
    
    X, y, scaler = prepare_lstm_data(data, look_back_days, future_days)
    
    if len(X) == 0:
        print(f"  Not enough data")
        return None

    model = build_lstm_model(look_back_days, future_days, lstm_units, dropout)
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    last_sequence = X[-1:]
    pred_scaled = model.predict(last_sequence, verbose=0)[0]
    predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    print(f"  LSTM trained")
    return predictions


# =======================
# RNN MODEL (SimpleRNN)
# =======================

def build_rnn_model(look_back_days=60, future_days=5, rnn_units=50, dropout=0.2):
    """Build SimpleRNN model"""
    model = Sequential()
    model.add(SimpleRNN(rnn_units, return_sequences=True, input_shape=(look_back_days, 1)))
    model.add(Dropout(dropout))
    model.add(SimpleRNN(rnn_units))
    model.add(Dropout(dropout))
    model.add(Dense(future_days))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_rnn_model(data, future_days=5, look_back_days=60, rnn_units=50, dropout=0.2, epochs=10):
    """Train SimpleRNN model"""
    print(f"  Training RNN - units={rnn_units}, dropout={dropout}, epochs={epochs}...")
    
    X, y, scaler = prepare_lstm_data(data, look_back_days, future_days)
    
    if len(X) == 0:
        print(f" Not enough data")
        return None

    model = build_rnn_model(look_back_days, future_days, rnn_units, dropout)
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    last_sequence = X[-1:]
    pred_scaled = model.predict(last_sequence, verbose=0)[0]
    predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    print(f"RNN trained")
    return predictions


# =======================
# ENSEMBLE EXPERIMENTATION
# =======================

def run_ensemble_experimentation(data, ticker, future_days=5, look_back_days=60):
    """Test LSTM, RNN, ARIMA, and SARIMA configurations"""
    print(f"\n{'='*70}")
    print(f"ENSEMBLE EXPERIMENTATION - {ticker}")
    print(f"Testing LSTM, RNN, ARIMA, and SARIMA models")
    print(f"{'='*70}\n")
    
    results = {}
    prices = data['Close']
    
    # ARIMA Configurations
    print("1. Testing ARIMA Models")
    print("-" * 70)
    arima_configs = [
        {'order': (1, 1, 1), 'name': 'ARIMA (1,1,1)'},
        {'order': (2, 1, 1), 'name': 'ARIMA (2,1,1)'},
        {'order': (1, 1, 2), 'name': 'ARIMA (1,1,2)'},
    ]
    
    for config in arima_configs:
        pred = train_arima_model(prices, future_days, config['order'])
        if pred is not None:
            results[config['name']] = pred
    
    # SARIMA Configurations
    print("\n2. Testing SARIMA Models")
    print("-" * 70)
    sarima_configs = [
        {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 5), 'name': 'SARIMA (1,1,1)x(1,1,1,5)'},
        {'order': (2, 1, 1), 'seasonal_order': (1, 1, 1, 5), 'name': 'SARIMA (2,1,1)x(1,1,1,5)'},
    ]
    
    for config in sarima_configs:
        pred = train_sarima_model(prices, future_days, config['order'], config['seasonal_order'])
        if pred is not None:
            results[config['name']] = pred
    
    # LSTM Configurations
    print("\n3. Testing LSTM Models")
    print("-" * 70)
    lstm_configs = [
        {'lstm_units': 30, 'dropout': 0.2, 'epochs': 5, 'name': 'LSTM (30 units, 5 epochs)'},
        {'lstm_units': 50, 'dropout': 0.2, 'epochs': 10, 'name': 'LSTM (50 units, 10 epochs)'},
    ]
    
    for config in lstm_configs:
        pred = train_lstm_model(data, future_days, look_back_days, 
                               config['lstm_units'], config['dropout'], config['epochs'])
        if pred is not None:
            results[config['name']] = pred
    
    # RNN Configurations
    print("\n4. Testing RNN Models")
    print("-" * 70)
    rnn_configs = [
        {'rnn_units': 30, 'dropout': 0.2, 'epochs': 5, 'name': 'RNN (30 units, 5 epochs)'},
        {'rnn_units': 50, 'dropout': 0.2, 'epochs': 10, 'name': 'RNN (50 units, 10 epochs)'},
    ]
    
    for config in rnn_configs:
        pred = train_rnn_model(data, future_days, look_back_days, 
                              config['rnn_units'], config['dropout'], config['epochs'])
        if pred is not None:
            results[config['name']] = pred
    
    # Create Ensemble Predictions
    print("\n5. Creating Ensemble Combinations")
    print("-" * 70)
    
    if len(results) >= 2:
        all_predictions = np.array(list(results.values()))
        
        ensemble_avg = np.mean(all_predictions, axis=0)
        results['ENSEMBLE_AVERAGE'] = ensemble_avg
        print("Ensemble Average created")
        
        ensemble_median = np.median(all_predictions, axis=0)
        results['ENSEMBLE_MEDIAN'] = ensemble_median
        print("Ensemble Median created")
    
    # Display Results
    print(f"\n{'='*70}")
    print("PREDICTION RESULTS")
    print(f"{'='*70}\n")
    
    for model_name, predictions in results.items():
        print(f"{model_name}:")
        for day in range(future_days):
            print(f"  Day {day+1}: ${predictions[day]:.2f}", end="  ")
        print("\n")
    
    # Plot Results
    plot_ensemble_comparison(results, ticker, future_days)
    
    return results


def plot_ensemble_comparison(results, ticker, future_days):
    """Visualize all ensemble predictions"""
    plt.figure(figsize=(16, 10))
    
    days = range(1, future_days + 1)
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    
    for idx, (model_name, predictions) in enumerate(results.items()):
        if 'ENSEMBLE' in model_name:
            plt.plot(days, predictions, marker='o', linewidth=3, 
                    label=model_name, color=colors[idx], markersize=10)
        else:
            plt.plot(days, predictions, marker='o', linewidth=1.5, 
                    label=model_name, color=colors[idx], alpha=0.7, markersize=6)
    
    plt.title(f'{ticker} - Ensemble Comparison (LSTM + RNN + ARIMA + SARIMA)', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Days Ahead', fontsize=12)
    plt.ylabel('Predicted Stock Price ($)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =======================
# MAIN MENU
# =======================

print("=== Stock Prediction v0.6 - Ensemble (LSTM + RNN + ARIMA + SARIMA) ===")
stock = input("Enter stock symbol (like AAPL, TSLA, GOOGL): ").upper() or "AAPL"
data = get_stock_data(stock)

if data is not None:
    print("\nWhat do you want to do?")
    print("1 = Make charts (candlestick + boxplot)")
    print("2 = Run ensemble experimentation (LSTM + RNN + ARIMA + SARIMA)")
    main_choice = input("Enter 1 or 2: ")

    if main_choice == "1":
        make_candlestick_chart(data, stock, days_per_candle=1)
        make_boxplot_chart(data, stock, window_size=5)

    elif main_choice == "2":
        future_days = int(input("How many days ahead to predict? (default=5): ") or "5")
        look_back = int(input("How many past days to use? (default=60): ") or "60")
        run_ensemble_experimentation(data, stock, future_days, look_back)

    else:
        print("Invalid choice!")
else:
    print("No data available. Exiting...")