# v0.3 - Stock Charts + Deep Learning Model Builder (Updated with Training)

# --- Libraries for stock charts ---
import yfinance as yf       # This gets stock data from Yahoo Finance
import pandas as pd         # This handles data tables
import mplfinance as mpf    # This makes candlestick charts only
import matplotlib.pyplot as plt  # This makes other types of charts

# --- Libraries for Deep Learning ---
from tensorflow.keras.models import Sequential   # Sequential model = layers added in order
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout  # Layers for DL models
import numpy as np  # For making dummy training data


def get_stock_data(ticker):
    """Get 2 years of stock data"""
    print(f"Getting {ticker} data...")  # Tell user what we're doing
    data = yf.download(ticker, period="2y")  # Download 2 years worth of stock data
    
    if data.empty:  # If we got no data back
        print(f"No data found for {ticker}")  # Tell user there's no data
        return None  # Return nothing
    
    # Fix multi-level columns if they exist
    if data.columns.nlevels > 1:  # Sometimes the data comes with weird column names
        data.columns = data.columns.droplevel(1)  # Fix the column names
    
    # Fix data types - make sure all numbers are actually numbers
    for column in ['Open', 'High', 'Low', 'Close', 'Volume']:  # Go through each column
        if column in data.columns:  # If this column exists
            data[column] = pd.to_numeric(data[column], errors='coerce')  # Convert to numbers
    
    # Remove any unnecessary rows
    data = data.dropna()  # Delete any rows with missing data
    
    print(f"Got {len(data)} days of data")  # Tell user how much data we got
    return data  # Give back the clean data

def group_days_simple(data, num_days):
    """Group daily data into weekly/monthly candles"""
    if num_days == 1:
        return data  # No grouping needed if daily
    
    grouped_data = []  # Empty list to store combined candles
    dates = []  # Empty list to store dates
    
    for i in range(0, len(data), num_days):  # Go through data in steps of num_days
        chunk = data.iloc[i:i+num_days]  # Get a chunk of days
        if len(chunk) > 0:
            combined = {
                'Open': chunk['Open'].iloc[0],        # First day's opening price
                'High': chunk['High'].max(),          # Highest price in the period
                'Low': chunk['Low'].min(),            # Lowest price in the period
                'Close': chunk['Close'].iloc[-1],     # Last day's closing price
                'Volume': chunk['Volume'].sum()       # Total volume
            }
            grouped_data.append(combined)
            dates.append(chunk.index[-1])  # Use last date in the group
    
    result = pd.DataFrame(grouped_data, index=dates)  # Make a new DataFrame
    return result

def make_candlestick_chart(data, ticker, days_per_candle=1):
    """Make a candlestick chart - daily, weekly, or monthly"""
    if data is None:
        print("No data to show!")
        return
    
    if days_per_candle > 1:
        print(f"Creating {days_per_candle}-day candles...")
        data = group_days_simple(data, days_per_candle)
    
    recent_data = data.tail(100)  # Show last 100 candles only
    
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

def make_boxplot_chart(data, ticker, window_size=5):
    """Make a boxplot using moving windows of closing prices"""
    if data is None:
        print("No data to show!")
        return
    
    print(f"Creating boxplot with {window_size}-day moving windows...")
    
    windows = []  # Store price windows
    window_labels = []  # Store labels for x-axis
    
    for i in range(len(data) - window_size + 1):
        window_data = data.iloc[i:i+window_size]
        prices = window_data['Close'].tolist()
        windows.append(prices)
        window_labels.append(str(window_data.index[-1].date()))
    
    if len(windows) > 50:
        step = len(windows) // 50  # Skip some to avoid clutter
        windows = windows[::step]
        window_labels = window_labels[::step]
    
    plt.figure(figsize=(14, 7))
    box_plot = plt.boxplot(windows, showmeans=True, patch_artist=True)
    for patch in box_plot['boxes']:
        patch.set_facecolor('lightblue')
    
    plt.title(f"{ticker} Stock Prices - {window_size}-Day Moving Window Boxplot")
    plt.xlabel(f"Trading Periods ({window_size} consecutive days each)")
    plt.ylabel("Closing Price ($)")
    
    tick_positions = range(1, len(window_labels) + 1, max(1, len(window_labels) // 10))
    tick_labels = [window_labels[i-1] for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Boxplot shows {len(windows)} windows of {window_size} days each")

# =======================
# v0.3 implementation, machine learning
# =======================

def build_model(layer_type="LSTM", layer_sizes=[50, 50], input_shape=(60, 1), dropout_rate=0.2, optimizer="adam", loss="mse"): # keras model where it lets the user pick between 'lstm', 'gru' or 'rnn'. layer_sizes is a list of how many neurons each layer should have by default its [50,50].
    """Build a Deep Learning model based on user input""" #input_shape will tell the first layer what shape the data has, (60 timesteps, 1 feature currently), dropout_rate represents the fraction of neurons to ignore during the training process, the optimiser and loss controls how ht emodel learns. 'adam' is the adaptive optimizer
    model = Sequential()  # Start a new sequential model
    LayerClass = {"LSTM": LSTM, "GRU": GRU, "RNN": SimpleRNN}[layer_type]  # Pick layer type
    
    for i, size in enumerate(layer_sizes):
        return_seq = (i < len(layer_sizes) - 1)  # True if not last layer
        if i == 0:
            model.add(LayerClass(size, return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(LayerClass(size, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    print(f"Built {layer_type} model with layers {layer_sizes}, dropout={dropout_rate}")
    return model

# =======================
# MAIN PROGRAM
# =======================

print("=== Stock Chart Maker v0.3 ===")

stock = input("Enter stock symbol (like AAPL, TSLA, GOOGL): ").upper()  # Ask for stock symbol
if not stock:
    stock = "AAPL"  # Default symbol

print("\nWhat chart do you want?")
print("1 = Candlestick chart")
print("2 = Boxplot chart")
choice = input("Enter 1 or 2: ")

data = get_stock_data(stock)

if choice == "2":
    try:
        window = int(input("Enter window size (default=5): ") or "5")
    except:
        window = 5
    make_boxplot_chart(data, stock, window)
else:
    print("\nChoose time period:")
    print("1 = Daily candles")
    print("5 = Weekly candles (5 days each)")
    print("20 = Monthly candles (20 days each)")
    try:
        days = int(input("Enter number (default=1): ") or "1")
    except:
        days = 1
    make_candlestick_chart(data, stock, days)

# =======================
# Training part
# =======================

print("\n=== Deep Learning Model Builder Demo ===")
demo_model = build_model(layer_type="LSTM", layer_sizes=[50, 50], input_shape=(60, 1))
demo_model.summary()  # Show model architecture

# make dummy data to test training (1000 samples, 60 timesteps, 1 feature) 
X_dummy = np.random.random((1000, 60, 1))
y_dummy = np.random.random((1000, 1))

# Ask user for training hyperparameters 
epochs = int(input("Enter number of epochs (default=5): ") or "5")
batch_size = int(input("Enter batch size (default=32): ") or "32")

# Train the model
print(f"\nTraining model for {epochs} epochs with batch size {batch_size}...")
history = demo_model.fit(X_dummy, y_dummy, epochs=epochs, batch_size=batch_size, verbose=1)

print("Training complete.")
print("Done")