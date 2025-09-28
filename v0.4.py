# === Stock Chart + Deep Learning Prediction Tool v0.4 ===

# --- Libraries for stock charts ---
import yfinance as yf                 # Download stock data from Yahoo Finance
import pandas as pd                  # Handle tabular data
import mplfinance as mpf             # Make candlestick charts
import matplotlib.pyplot as plt      # Make other types of charts (like boxplots)

# --- Libraries for Deep Learning ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# =======================
# DATA FETCHING + CHARTS
# =======================

def get_stock_data(ticker):
    """Download 2 years of stock data and clean it up"""
    print(f"Getting {ticker} data...")
    data = yf.download(ticker, period="2y")  # Get 2 years of history

    if data.empty:
        print(f"No data found for {ticker}")
        return None

    if data.columns.nlevels > 1:
        data.columns = data.columns.droplevel(1)

    # Convert columns to numeric and drop missing data
    for column in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']: #added Adj close as well 
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna()

    print(f"Got {len(data)} days of data")
    return data


def group_days_simple(data, num_days):
    """Group daily data into combined candles (e.g. weekly or monthly)"""
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
    """Show candlestick chart (daily, weekly, or monthly)"""
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

    print(f"Making {period_name.lower()} chart with {len(recent_data)} candles...")
    mpf.plot(
        recent_data,
        type='candle',
        title=f'{ticker} Stock Price ({period_name})',
        ylabel='Price ($)',
        volume=True,
        style='charles',
        figsize=(12, 6)
    )
    plt.show()


def make_boxplot_chart(data, ticker, window_size=5):
    """Make a boxplot using moving windows of closing prices"""
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
    print(f"Boxplot shows {len(windows)} windows of {window_size} days each")


# =======================
# MULTISTEP PREDICTION (SINGLE FEATURE - CLOSING PRICE)
# =======================

def prepare_data_for_multistep(data, look_back_days=60, future_days=5):
    """Prepare closing prices for multi-day prediction"""
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


def build_multistep_model(look_back_days=60, future_days=5):
    """Build a simple LSTM model for multi-day prediction"""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back_days, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(future_days))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_and_predict_multistep(data, ticker, look_back_days=60, future_days=5, epochs=20):
    """Predict multiple days of closing prices using only closing price history"""
    print(f"\n=== Multistep Prediction ===")
    print(f"Predicting {future_days} days of closing prices using only past closing prices")
    
    X, y, scaler = prepare_data_for_multistep(data, look_back_days, future_days)
    if len(X) == 0:
        print("Not enough data for training!")
        return

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    
    model = build_multistep_model(look_back_days, future_days)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    predictions = model.predict(X_test)
    predictions_real = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    plot_multistep_results(predictions_real, y_test_real, ticker, future_days, "Multistep (Close Only)")
    return model, scaler


# =======================
#MULTIVARIATE PREDICTION (ALL FEATURES, SINGLE DAY)
# =======================

def prepare_data_for_multivariate(data, look_back_days=60):
    """Prepare all features (Open, High, Low, Close, Adj Close, Volume) to predict next day's closing price"""
    # Use all available features
    possible_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    feature_columns = [col for col in possible_columns if col in data.columns]
    
    print(f"Available features: {feature_columns}")
    if len(feature_columns) < 4:
        print("Warning: Less than 4 features available. Results may not be optimal.")
    
    features = data[feature_columns].values
    
    # Scale the features
    scaler_features = MinMaxScaler()
    scaled_features = scaler_features.fit_transform(features)
    
    # Scale the target (closing price) separately
    target = data['Close'].values.reshape(-1, 1)
    scaler_target = MinMaxScaler()
    scaled_target = scaler_target.fit_transform(target)

    X, y = [], []
    for i in range(look_back_days, len(scaled_features)):
        X.append(scaled_features[i-look_back_days:i])  # All features for past days
        y.append(scaled_target[i, 0])  # Next day's closing price

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler_features, scaler_target


def build_multivariate_model(look_back_days=60, num_features=6):
    """Build LSTM model for multivariate single-step prediction"""
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(look_back_days, num_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Predict single value (next day's closing price)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_and_predict_multivariate(data, ticker, look_back_days=60, epochs=20):
    """need to predict next day's closing price using all stock features"""
    print(f"\n=== Multivariate Prediction ===")
    print("Predicting next day's closing price using available features (Open, High, Low, Close, Adj Close, Volume)")
    
    X, y, scaler_features, scaler_target = prepare_data_for_multivariate(data, look_back_days)
    if len(X) == 0:
        print("Not enough data for training!")
        return

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    print(f"Using {X.shape[2]} features")
    
    model = build_multivariate_model(look_back_days, X.shape[2])  # Use actual number of features
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    predictions = model.predict(X_test)
    predictions_real = scaler_target.inverse_transform(predictions)
    y_test_real = scaler_target.inverse_transform(y_test.reshape(-1, 1))

    plot_single_step_results(predictions_real, y_test_real, ticker, "Multivariate (Available Features)")
    return model, scaler_features, scaler_target


def plot_single_step_results(predictions, actuals, ticker, method_name):
    """Plot single-step prediction results"""
    plt.figure(figsize=(14, 6))
    
    # Plot first 100 predictions vs actuals
    sample_size = min(100, len(predictions))
    days = range(sample_size)
    
    plt.subplot(1, 2, 1)
    plt.plot(days, actuals[:sample_size], 'b-', label='Actual', alpha=0.7)
    plt.plot(days, predictions[:sample_size], 'r-', label='Predicted', alpha=0.7)
    plt.title(f'{ticker} - {method_name}\nFirst {sample_size} Test Predictions')
    plt.xlabel('Test Sample')
    plt.ylabel('Closing Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot to show correlation
    plt.subplot(1, 2, 2)
    plt.scatter(actuals[:sample_size], predictions[:sample_size], alpha=0.6)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and show accuracy metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    print(f"Results for {method_name}:")
    print(f"  Root Mean Square Error (RMSE): ${rmse:.2f}")
    print(f"  Mean Absolute Error (MAE): ${mae:.2f}")


# =======================
# MULTIVARIATE MULTISTEP PREDICTION (ALL FEATURES, MULTIPLE DAYS)
# =======================

def prepare_data_for_multivariate_multistep(data, look_back_days=60, future_days=5):
    """Prepare all features to predict multiple days of closing prices"""
    # Use all available features for input - check which ones exist
    possible_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    feature_columns = [col for col in possible_columns if col in data.columns]
    
    print(f"Available features: {feature_columns}")
    if len(feature_columns) < 4:
        print("Warning: Less than 4 features available. Results may not be optimal.")
    
    features = data[feature_columns].values
    
    # Scale the features
    scaler_features = MinMaxScaler()
    scaled_features = scaler_features.fit_transform(features)
    
    # Scale the target (closing price) separately
    target = data['Close'].values.reshape(-1, 1)
    scaler_target = MinMaxScaler()
    scaled_target = scaler_target.fit_transform(target)

    X, y = [], []
    for i in range(look_back_days, len(scaled_features) - future_days + 1):
        X.append(scaled_features[i-look_back_days:i])  # All features for past days
        y.append(scaled_target[i:i+future_days, 0])  # Multiple future closing prices

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler_features, scaler_target


def build_multivariate_multistep_model(look_back_days=60, num_features=6, future_days=5):
    """Build LSTM model for multivariate multistep prediction"""
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(look_back_days, num_features)))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(future_days))  # Predict multiple days
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_and_predict_multivariate_multistep(data, ticker, look_back_days=60, future_days=5, epochs=20):
    """ Predict multiple days of closing prices using all stock features"""
    print(f"\n=== Multivariate Multistep Prediction ===")
    print(f"Predicting {future_days} days of closing prices using available features")
    
    X, y, scaler_features, scaler_target = prepare_data_for_multivariate_multistep(data, look_back_days, future_days)
    if len(X) == 0:
        print("Not enough data for training!")
        return

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    print(f"Using {X.shape[2]} features")
    
    model = build_multivariate_multistep_model(look_back_days, X.shape[2], future_days)  # Use actual number of features
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    predictions = model.predict(X_test)
    predictions_real = scaler_target.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    y_test_real = scaler_target.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    plot_multistep_results(predictions_real, y_test_real, ticker, future_days, "Multivariate Multistep (Available Features)")
    return model, scaler_features, scaler_target


def plot_multistep_results(predictions, actuals, ticker, future_days, method_name):
    """Show first 5 prediction samples for multistep results"""
    plt.figure(figsize=(15, 8))
    for i in range(min(5, len(predictions))):
        plt.subplot(2, 3, i+1)
        days = range(1, future_days + 1)
        plt.plot(days, actuals[i], 'bo-', label='Actual', linewidth=2, markersize=6)
        plt.plot(days, predictions[i], 'ro-', label='Predicted', linewidth=2, markersize=6)
        plt.title(f'Test Sample {i+1}')
        plt.xlabel('Days Ahead')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'{ticker} - {method_name}\nPredicting {future_days} Days Ahead', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Calculate accuracy metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    print(f"Results for {method_name}:")
    print(f"  Root Mean Square Error (RMSE): ${rmse:.2f}")
    print(f"  Mean Absolute Error (MAE): ${mae:.2f}")


# =======================
# MAIN MENU
# =======================

print("=== Stock Chart + ML Predictor v0.4 ===")
stock = input("Enter stock symbol (like AAPL, TSLA, GOOGL): ").upper() or "AAPL"
data = get_stock_data(stock)

if data is not None:
    print("\nWhat do you want to do?")
    print("1 = Make charts (candlestick + boxplot)")
    print("2 = Requirement #1: Multistep prediction (Close price only)")
    print("3 = Requirement #2: Multivariate prediction (All features, 1 day)")
    print("4 = Requirement #3: Multivariate multistep (All features, multiple days)")
    main_choice = input("Enter 1, 2, 3, or 4: ")

    if main_choice == "1":
        make_candlestick_chart(data, stock, days_per_candle=1)
        make_boxplot_chart(data, stock, window_size=5)

    elif main_choice == "2":
        future_days = int(input("How many days ahead to predict? (default=5): ") or "5")
        look_back = int(input("How many past days to use? (default=60): ") or "60")
        epochs = int(input("How many training epochs? (default=20): ") or "20")
        train_and_predict_multistep(data, stock, look_back_days=look_back, future_days=future_days, epochs=epochs)

    elif main_choice == "3":
        look_back = int(input("How many past days to use? (default=60): ") or "60")
        epochs = int(input("How many training epochs? (default=20): ") or "20")
        train_and_predict_multivariate(data, stock, look_back_days=look_back, epochs=epochs)

    elif main_choice == "4":
        future_days = int(input("How many days ahead to predict? (default=5): ") or "5")
        look_back = int(input("How many past days to use? (default=60): ") or "60")
        epochs = int(input("How many training epochs? (default=20): ") or "20")
        train_and_predict_multivariate_multistep(data, stock, look_back_days=look_back, future_days=future_days, epochs=epochs)

    else:
        print("Invalid choice!")
else:
    print("No data available. Exiting...")