import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------
# 1. Load Data
# ----------------------------
def get_stock_data(ticker, start="2015-01-01", end="2025-01-01"):
    """
    Downloads stock data using yfinance.
    Args:
        ticker (str): Stock ticker symbol, e.g., "AAPL", "CBA.AX".
        start (str): Start date (YYYY-MM-DD).
        end (str): End date (YYYY-MM-DD).
    Returns:
        pd.DataFrame: Historical stock data.
    """
    try:
        # Creates filename using the ticket and dates specified by user for download
        filename = f"{ticker}_{start}_{end}.csv"
        
        # Check if file already exists on computer
        if os.path.exists(filename):
            print(f"Loading {ticker} data from local file: {filename}")
            df = pd.read_csv(filename, index_col=0, parse_dates=True) # pd.read_csv will load CSV data into a padas DataFrame, index_col=0 sets the first column which is dates as the data frame
            # parse_dates=True will convert date strings to datetime objects
        else:
            print(f"Downloading {ticker} data from internet...")
            df = yf.download(ticker, start=start, end=end) #yf.download() will get the past data from Yahoo Finance, It will Retrun a DataFrame with Open High Low Close Volume Data
            # Save to local file for next time
            # to_csv() exports DataFrame to CSV format
            df.to_csv(filename)
            print(f"Data saved to local file: {filename}")
        
        # ----------------------------
        # Trying to deal wit NaN
        # ----------------------------
        # Check if there are any missing values
        # df.isnull() will return Booldean values, true if there is 
        # .sum().sum() will add up the Nan from each column, then from each column
        print(f"NaN values in original data: {df.isnull().sum().sum()}")
        
        # make a NaN to test
        # df.iloc[row_index, column index] it will go into a the specific DataFrame Cell
        # df.columns.get_loc('Close') should return the integer positon of the close column
        df.iloc[2, df.columns.get_loc('Close')] = np.nan  
        print("Created 1 missing value for testing..")
        print(f"NaN values after test: {df.isnull().sum().sum()}")
        
        # Fix missing values
        df.ffill(inplace=True)  # forward fill - use last known price, inplace=True will modify the original DataFrame instead of creating a copy
        df.bfill(inplace=True)  # backward fill - for any remaining NaN from the beginning that might have been missed
        print(f"NaN values after fixing: {df.isnull().sum().sum()}")
        print("NaN handling works\n")
        
        return df
    except Exception as e:
        # Takes care of any errors throughout the download or processing
        print(f"Failed to download data for {ticker}: {e}")
        return pd.DataFrame() # Returns an empty DataFrame if it fails

# ----------------------------
# 2. Split Data
# ----------------------------
#splits 80/20 based on chronlogical order
def split_data(df):

    # Calculates the point of split as 80% of the total data length
    # int() converts float result to ineger for array indexing
    split_point = int(len(df) * 0.8)  # 80% for training

    #used iloc for integer location indexing, [:split_point] selects the bgining up to but not incuding split point
    train_df = df.iloc[:split_point]   # First 80%
    test_df = df.iloc[split_point:]    # Last 20% - [:split_point] selects from split_point to the end
    print(f"Data split: {len(train_df)} for training, {len(test_df)} for testing")
    return train_df, test_df

# ----------------------------
# 3. Data Processing
# ----------------------------
def prepare_data(df):
    """
    Prepares data for training: scales values and creates features.
    Args:
        df (pd.DataFrame): Stock price DataFrame.
    Returns:
        X (np.array): Features (day index).
        y (np.array): Target values (Close prices).
    """
    df = df.copy() # create a copy to avoid modifying the original Dataframe, .copy() creates a copy stopping changes that weren't supposed to happen
    df = df.reset_index()  # reset index to convert datetime to a regular column, it basically moves the date infomration from index to a column named as Date
    df["Days"] = np.arange(len(df)) #creates a day counter to start from 0, np.arrange(len(df)) generates [0,1,2.. len(df)-1], this acts as the time based feature for the model
    X = df[["Days"]].values # extract features for machine learning, uses double brackets so its a 2D array for sklearn, .values will convert pandas DataFrame to numpy array
    y = df["Close"].values # extrct the target vairable, which is what we want to predict, single brackets htis time retyrbs a Series, .values will convert to 1D numpy array
    return X, y, df

# ----------------------------
# 4. Train Model
# ----------------------------
def train_model(X, y):
    """
    Trains a Linear Regression model.
    """
    #initalise the model
    # creates an instance of LinearRegression class that has default parameters
    model = LinearRegression()
    model.fit(X, y) # train the model using fit, .fit() will calualte the the coefficents (slope and intercepts) using the least squares, difficult to write out the math symbols here, it also finds th ebest line that goes thorugh most points
    preds = model.predict(X) # see how it perfoms by predicting on same training data
    score = r2_score(y, preds) #r2 score tells how good the fit is (1 is perfect), basically shows if the line explains the price changes well enough
    return model, score

# ----------------------------
# 5. Predict Future
# ----------------------------
def predict_future(model, df, days_ahead=30):
    """
    Predicts stock price for future days.
    """
    # Find what the last day number was in the data
    last_day = df["Days"].max()
    # Creates an array of the future day numbers, it will start from the day after the last_day, reshape(-1,1) turns it into column format that sklean uses, -1 will mean that it figures out how many rows automatically 
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1).reshape(-1, 1) # with the model thats trained predict the prices futuee days, pretty much extends the line that was created in training
    future_preds = model.predict(future_days)
    
    
    future_df = pd.DataFrame({ #puts info into a table
        "Days": future_days.flatten(), # .flatten() turns the array back into the simpe lists for the DataFrame
        "Predicted": future_preds.flatten()
    })
    return future_df

# ----------------------------
# 6. Plot Results
# ----------------------------
def plot_results(df, future_df, model):
    """
    Plots historical data, model fit, and future predictions.
    """
    plt.figure(figsize=(10,6))
    
    plt.plot(df.index, df["Close"], label="Historical Close")
    
    # For the model fit, we need to prepare the data first
    _, _, df_prepared = prepare_data(df)
    plt.plot(df.index, model.predict(df_prepared[["Days"]]), label="Model Fit", linestyle="--")
    
    # Extend future dates
    future_dates = pd.date_range(start=df.index[-1], periods=len(future_df)+1, freq="B")[1:]
    plt.plot(future_dates, future_df["Predicted"], label="Future Prediction", color="green")
    
    plt.title("Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

# ----------------------------
# 7. Main
# ----------------------------
if __name__ == "__main__":
    # Ask user for inputs, fall back to defaults if empty
    # input() function will pause execution and wait for the user's input, 'or' operator provides deafult values if the user presses enter and tries continuing without filling it in
    ticker = input("Enter stock ticker (e.g., CBA.AX, AAPL, TSLA): ") or "CBA.AX"
    start = input("Enter start date (YYYY-MM-DD): ") or "2015-01-01"
    end = input("Enter end date (YYYY-MM-DD): ") or "2025-01-01"

    print(f"Downloading data for {ticker} from {start} to {end}...")

    #Checks if data download was completed
    # .empty returns True if DataFrame has no data (0 rows or 0 columns)
    df = get_stock_data(ticker, start=start, end=end)
    if df.empty:
        print("No data found. Check ticker symbol or date range.")
    else:
        # Split the data (simple 80/20 split)
        train_df, test_df = split_data(df)
        
        # Train on training data
        #converts the raw price data into a feature matrix and target vector
        X_train, y_train, train_df = prepare_data(train_df)
        # train the model bnased on prepared training data
        #Will return the trained model object and the performance score
        model, train_score = train_model(X_train, y_train)

        #Show the model perofrmance using R^2 score (aiming for 1)
        # .2f will keep it to 2 decmial places
        print(f"Model trained. R² Score: {train_score:.2f}")

        # Predict future
        # prepares the full dataset for prediction visualisation, use full data not just training to show complete model performance
        X_full, y_full, df_full = prepare_data(df)
        future_df = predict_future(model, df_full, days_ahead=30) #days_ahead= 30 predicts the prices for the next 30 business days.
        print("First few predictions:\n", future_df.head()) # shows the user the first few predictions, .head() shwos the first 5 rows by default

        #plot the historical data, model fit and future prediction in one plot
        plot_results(df, future_df, model)