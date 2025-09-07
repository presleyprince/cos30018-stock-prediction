# Import the libraries we need
import yfinance as yf       # This gets stock data from Yahoo Finance
import pandas as pd         # This handles data tables
import mplfinance as mpf    # This makes candlestick charts only
import matplotlib.pyplot as plt  # This makes other types of charts

def get_stock_data(ticker):
    """Get 2 years of stock data"""
    print(f"Getting {ticker} data...")  # Tell user what we're doing
    data = yf.download(ticker, period="2y")  # Download 2 years worth of stock data
    
    if data.empty:  # If we got no data back
        print(f"No data found for {ticker}")  # Tell user there's no data
        return None  # Return nothing
    
    # Fix multi-level columns if they exist, Multi-level columns happen when Yahoo Finance returns column names e.g ('Open', 'AAPL') and not just single 'Open', it happens because of formatting issues while downloading.
    if data.columns.nlevels > 1:  # Sometimes the data comes with weird column names
        data.columns = data.columns.droplevel(1)  # Fix the column names, fixes multi-level columns by getting rid of the second part of the column name so the chart function can just use data['Open'] insead of something complicated like data[('Open', 'AAPL')]
    
    # Fix data types - make sure all numbers are actually numbers
    for column in ['Open', 'High', 'Low', 'Close', 'Volume']:  # Go through each column
        if column in data.columns:  # If this column exists
            data[column] = pd.to_numeric(data[column], errors='coerce')  # Make sure it's a number, it will loop through each column one nby one, it will make sure the column does exist, it will then convert it to numbers 'pd.to_numeric()' means turn this into numbers, 'errors='coerce'' if it can't convert something make it into NaN rather than crashing.
    
    # Remove any unnecessary rows
    data = data.dropna()  # Delete any rows with missing data, if theres any NaN's removes that row.
    
    print(f"Got {len(data)} days of data")  # Tell user how much data we got, counts how many rows(days) are in the data table
    return data  # Give back the clean data, if return data wasn't there the other functions would have nothing to make charts with is what i found with errors.

def group_days_simple(data, num_days): # function will combine multiple days into single candles (5 daily candles to 1 weekly candle)
    """Group daily data into weekly/monthly candles"""
    if num_days == 1:  # If they want daily data
        return data  # Just give back the original data, since no need to group anything
    
    grouped_data = []  # Empty list to store combined candles, Open, High, Low, CLose, Volume (The prices)
    dates = []  # Empty list to store dates, dates for each combined candle (the dates the prices belong to)
    
    # Go through data in chunks of num_days
    for i in range(0, len(data), num_days):  # Jump through data in steps of num_days, 5 numdays = weekly, 20 numdays = monthly chunks.
        chunk = data.iloc[i:i+num_days]  # Get a chunk of days
        if len(chunk) > 0:  # If the chunk has data
            # Combine the chunk into one candle
            combined = {
                'Open': chunk['Open'].iloc[0],        # First day's opening price, iloc is pandas way of finding positions
                'High': chunk['High'].max(),          # Highest price in the period
                'Low': chunk['Low'].min(),            # Lowest price in the period
                'Close': chunk['Close'].iloc[-1],     # Last day's closing price, the negative one will count backwards from the end to get the last.
                'Volume': chunk['Volume'].sum()       # Total volume for all days
            }
            grouped_data.append(combined)  # Add this combined candle to our list, combined is the single, weekly/monthly candle made from multiple days, appened will add it to the end of the list
            dates.append(chunk.index[-1])  # Remember the last date in this period, adds the last data to the dates list
    
    # Create new DataFrame
    result = pd.DataFrame(grouped_data, index=dates)  # Turn our list into a proper data table (so pandas can wor with it), index = dates uses the date list as the row labels 
    return result  # Give back the grouped data, formatted data table

def make_candlestick_chart(data, ticker, days_per_candle=1):
    """Make a candlestick chart - daily, weekly, or monthly"""
    if data is None:  # If we have no data
        print("No data to show!")  # Tell user
        return  # Stop here
    
    # Group data if needed
    if days_per_candle > 1:  # If they want weekly or monthly candles
        print(f"Creating {days_per_candle}-day candles...")  # Tell user what we're doing
        data = group_days_simple(data, days_per_candle)  # Group the days together
    
    # Only show last 100 candles so chart isn't too crowded
    recent_data = data.tail(100)  # Take only the last 100 candles
    
    period_name = "Daily"  # Default name
    if days_per_candle == 5:  # If it's 5 days
        period_name = "Weekly"  # Call it weekly
    elif days_per_candle == 20:  # If it's 20 days
        period_name = "Monthly"  # Call it monthly
    
    print(f"Making {period_name.lower()} chart with {len(recent_data)} candles...")  # Tell user what we're making
    
    # Make the chart
    mpf.plot(
        recent_data,  # The data to show
        type='candle',  # Make it a candlestick chart
        title=f'{ticker} Stock Price ({period_name})',  # Chart title
        ylabel='Price ($)',  # Label for the price axis
        volume=True,  # Show volume bars at bottom
        style='charles',  # Color scheme
        figsize=(12, 6)  # Size of the chart
    )

def make_boxplot_chart(data, ticker, window_size=5):
    """
    Display stock market financial data using boxplot chart.
    This is particularly useful when you are trying to display your data 
    for a moving window of n consecutive trading days for the boxplot.
    """
    if data is None:  # If we have no data
        print("No data to show!")  # Tell user
        return  # Stop here
    
    print(f"Creating boxplot with {window_size}-day moving windows...")  # Tell user what we're doing
    
    # Create moving windows of n consecutive trading days
    windows = []  # Empty list for all our windows, stores the data for each boxplot
    window_labels = []  # Empty list for date labels, stores the dates for labelling each box on the x-axis
    
    # Slide the window across all the data
    for i in range(len(data) - window_size + 1):  # Move the window one day at a time
        window_data = data.iloc[i:i+window_size]  # Get n consecutive days
        prices = window_data['Close'].tolist()    # Get just the closing prices
        windows.append(prices)  # Add these prices to our list
        # Label with the end date of the window
        window_labels.append(str(window_data.index[-1].date()))  # Remember the last date
    
    # Only show every 10th window so chart isn't too crowded
    if len(windows) > 50:  # If we have too many windows
        step = len(windows) // 50  # Calculate how many to skip, e.g. 200 windows = 200/50 = 4 meaning keep every 4th window and skip the rest
        windows = windows[::step]  # Keep every window in intervals, using python slicing, start at index 0 and go to the end of the list, take every certain amount after being skipped, e.g. step = 3 [1-10], first element = 1 then 2nd element is 4 third element is 7, then stop
        window_labels = window_labels[::step]  # Not showing every label just some of them at regular intervales
    
    # Make the boxplot
    plt.figure(figsize=(14, 7))  # Create a chart area
    box_plot = plt.boxplot(windows, showmeans=True, patch_artist=True)  # Make the boxplot
    
    # Color the boxes
    for patch in box_plot['boxes']:  # For each box in the chart
        patch.set_facecolor('lightblue')  # Make it light blue
    
    plt.title(f"{ticker} Stock Prices - {window_size}-Day Moving Window Boxplot")  # Chart title
    plt.xlabel(f"Trading Periods ({window_size} consecutive days each)")  # X-axis label
    plt.ylabel("Closing Price ($)")  # Y-axis label
    
    # Show every certain amount of labels to avoid crowding (usually roughly every 10th label)
    tick_positions = range(1, len(window_labels) + 1, max(1, len(window_labels) // 10))  # Calculate which labels to show
    tick_labels = [window_labels[i-1] for i in tick_positions]  # Get those labels
    plt.xticks(tick_positions, tick_labels, rotation=45)  # Put labels on x-axis, rotated 45 degrees
    
    plt.grid(True, alpha=0.3)  # Add a faint grid
    plt.tight_layout()  # Make everything fit nicely
    plt.show()  # Show the chart
    
    print(f"Boxplot shows {len(windows)} windows of {window_size} consecutive trading days each")  # Tell user what we made

# Main program
print("=== Stock Chart Maker ===")  

# Get stock symbol from user
stock = input("Enter stock symbol (like AAPL, TSLA, GOOGL): ").upper()  # Ask user for stock symbol and make it uppercase
if not stock:  # If user didn't type anything
    stock = "AAPL"  # Use Apple as default

# Ask what kind of chart
print("\nWhat chart do you want?")  # Ask user what they want
print("1 = Candlestick chart")  # Option 1
print("2 = Boxplot chart")  # Option 2
choice = input("Enter 1 or 2: ")  # Get user's choice

# Get the data
data = get_stock_data(stock)  # Download the stock data

# Make the chosen chart
if choice == "2":  # If they chose boxplot
    # Boxplot - ask for window size
    try:  # Try to get window size
        window = int(input("Enter window size (consecutive trading days, default=5): ") or "5")  # Ask for window size
    except:  # If they enter something bad
        window = 5  # Use 5 as default
    make_boxplot_chart(data, stock, window)  # Make the boxplot
else:  # Otherwise (candlestick)
    # Candlestick - ask for time period
    print("\nChoose time period:")  # Ask for time period
    print("1 = Daily candles")  # Daily option
    print("5 = Weekly candles (5 days each)")  # Weekly option
    print("20 = Monthly candles (20 days each)")  # Monthly option
    try:  # Try to get their choice
        days = int(input("Enter number (default=1): ") or "1")  # Ask for number of days
    except:  # If they enter something bad
        days = 1  # Use daily as default
    make_candlestick_chart(data, stock, days)  # Make the candlestick chart

print("Done") 