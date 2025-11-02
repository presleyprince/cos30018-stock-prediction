# ==============================================================================
# STOCK PREDICTION TOOL v0.6
# ==============================================================================

#imports
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

from newsapi import NewsApiClient
from datetime import datetime, timedelta
import os
import re

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Classification imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False
    print("Note: transformers not installed. FinBERT will not be available.")

# =======================
# SECTION 1: DATA FETCHING
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
# SECTION 2: NEWS COLLECTION
# =======================

def get_company_name(ticker):
    """Convert stock ticker to company name for better search results"""
    companies = {
        'AAPL': 'Apple', 'TSLA': 'Tesla', 'GOOGL': 'Google',
        'MSFT': 'Microsoft', 'AMZN': 'Amazon', 'META': 'Meta OR Facebook',
        'NVDA': 'NVIDIA', 'NFLX': 'Netflix', 'AMD': 'AMD', 'INTC': 'Intel'
    }
    return companies.get(ticker, ticker)


def clean_text_simple(text):
    """Clean up news text by removing junk"""
    if not text:
        return ""
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = ' '.join(text.split())
    return text.strip()


def fetch_news_simple(ticker, start_date, end_date, api_key):
    """Fetch news articles for a stock ticker"""
    print(f"Fetching news for {ticker}")
    
    today = datetime.now()
    max_days_back = 30
    earliest_allowed = today - timedelta(days=max_days_back)
    
    if start_date < earliest_allowed:
        print(f"Note that, free NewsAPI only allows last {max_days_back} days")
        print(f" Adjusting start date from {start_date.date()} to {earliest_allowed.date()}")
        start_date = earliest_allowed
    
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    newsapi = NewsApiClient(api_key=api_key)
    company_name = get_company_name(ticker)
    print(f"Searching for: {company_name}")
    
    all_articles = []
    current_date = start_date
    
    while current_date < end_date:
        chunk_end = min(current_date + timedelta(days=30), end_date)
        print(f"\nFetching: {current_date.date()} to {chunk_end.date()}...", end=" ")
        
        try:
            response = newsapi.get_everything(
                q=company_name,
                from_param=current_date.strftime('%Y-%m-%d'),
                to=chunk_end.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            if response['status'] == 'ok':
                articles = response['articles']
                all_articles.extend(articles)
                print(f"Found {len(articles)} articles")
            else:
                print("No articles found")
        except Exception as e:
            print(f"Error: {e}")
        
        current_date = chunk_end + timedelta(days=1)
    

    print(f"Total Articles collected: {len(all_articles)}")

    
    return all_articles


def process_articles_simple(articles, ticker):
    """Clean and organize the news articles into a DataFrame"""
    print("\nProcessing articles...")
    
    processed_data = []
    
    for article in articles:
        try:
            published_at = article.get('publishedAt', '')
            title = article.get('title', '')
            description = article.get('description', '')
            source = article.get('source', {}).get('name', 'Unknown')
            url = article.get('url', '')
            
            if not title or not published_at:
                continue
            
            pub_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
            date_only = pub_date.date()
            
            full_text = f"{title}. {description}" if description else title
            full_text = clean_text_simple(full_text)
            
            if len(full_text) < 20:
                continue
            
            processed_data.append({
                'date': date_only,
                'title': title,
                'description': description,
                'text': full_text,
                'source': source,
                'url': url
            })
        except:
            continue
    
    df = pd.DataFrame(processed_data)
    
    print(f"Processed {len(df)} articles")
    if len(df) > 0:
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Unique sources: {df['source'].nunique()}")
    
    return df


def remove_duplicates(df):
    """Remove duplicate articles"""
    print("\nRemoving duplicates...", end=" ")
    original_count = len(df)
    df = df.drop_duplicates(subset=['date', 'title'], keep='first')
    removed = original_count - len(df)
    print(f"Removed {removed} duplicates")
    return df


def align_news_with_trading_days(stock_data, news_df):
    """Match news articles with stock trading days"""
    print("\nAligning news with trading days...")
    
    trading_dates = sorted([d.date() for d in stock_data.index])
    print(f"  Trading days: {len(trading_dates)}")
    
    news_df['date'] = pd.to_datetime(news_df['date'])
    
    aligned_data = []
    
    for trade_date in trading_dates:
        news_for_day = []
        
        for days_back in range(1, 4):
            check_date = trade_date - timedelta(days=days_back)
            day_news = news_df[news_df['date'].dt.date == check_date]
            
            if len(day_news) > 0:
                news_for_day.append(day_news)
        
        if news_for_day:
            combined = pd.concat(news_for_day)
            all_text = ' '.join(combined['text'].values)
            
            aligned_data.append({
                'trading_date': trade_date,
                'article_count': len(combined),
                'combined_text': all_text
            })
    
    aligned_df = pd.DataFrame(aligned_data)
    
    print(f"Aligned news for {len(aligned_df)} trading days")
    print(f" Coverage: {len(aligned_df)/len(trading_dates)*100:.1f}% of trading days")
    
    return aligned_df


def save_news_data(df, ticker, filename_suffix):
    """Save news data to CSV file"""
    filename = f"{ticker}_{filename_suffix}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved to: {filename}")
    return filename


def collect_all_news(ticker, stock_data, api_key):
    """Main function to collect and process all news data"""

    print(f"News collection workflow for {ticker}")

    
    raw_file = f"{ticker}_news_raw.csv"
    aligned_file = f"{ticker}_news_aligned.csv"
    
    if os.path.exists(raw_file) and os.path.exists(aligned_file):
        print("\nExisting news data found!")
        print(f"  Files: {raw_file}, {aligned_file}")
        
        use_existing = input("\nUse existing data? (y/n): ").lower()
        if use_existing == 'y':
            print("Loading existing data...")
            aligned = pd.read_csv(aligned_file)
            print(f"Loaded {len(aligned)} days of news data")
            return aligned
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"\nFree NewsAPI Limitation:")
    print(f"Only collecting last 30 days of news")
    
    articles = fetch_news_simple(ticker, start_date, end_date, api_key)
    
    if len(articles) == 0:
        print("\nNo articles found! Try a different stock or check your API key.")
        return None
    
    news_df = process_articles_simple(articles, ticker)
    news_df = remove_duplicates(news_df)
    save_news_data(news_df, ticker, 'news_raw')
    
    aligned_news = align_news_with_trading_days(stock_data, news_df)
    save_news_data(aligned_news, ticker, 'news_aligned')
    

    print("Collection Complete")

    print(f"Total articles: {len(news_df)}")
    print(f"Trading days with news: {len(aligned_news)}")
    
    return aligned_news


# =======================
# SECTION 3: SENTIMENT ANALYSIS
# =======================

def analyze_sentiment_textblob(text):
    """TextBlob sentiment analysis"""
    try:
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    except:
        return {'polarity': 0, 'subjectivity': 0}


def analyze_sentiment_vader(text):
    """VADER sentiment analysis"""
    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    except:
        return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1}


def analyze_sentiment_finbert(text):
    """FinBERT sentiment analysis"""
    if not TRANSFORMERS_AVAILABLE:
        return {'positive': 0, 'negative': 0, 'neutral': 1}
    
    try:
        finbert = pipeline("sentiment-analysis", 
                          model="ProsusAI/finbert",
                          tokenizer="ProsusAI/finbert")
        
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        result = finbert(text)[0]
        label = result['label'].lower()
        score = result['score']
        
        sentiment = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiment[label] = score
        
        return sentiment
    except Exception as e:
        return {'positive': 0, 'negative': 0, 'neutral': 1}


def apply_sentiment_analysis(aligned_news, method='all'):
    """Apply sentiment analysis to all news articles"""

    print("Sentiment Analysis")
    print(f"Analyzing {len(aligned_news)} days of news...")
    
    sentiment_df = aligned_news.copy()
    
    for idx, row in sentiment_df.iterrows():
        text = row['combined_text']
        
        print(f"\nDay {idx+1}/{len(sentiment_df)}: {row['trading_date']}", end=" ")
        
        if method in ['textblob', 'all']:
            tb_scores = analyze_sentiment_textblob(text)
            sentiment_df.at[idx, 'textblob_polarity'] = tb_scores['polarity']
            sentiment_df.at[idx, 'textblob_subjectivity'] = tb_scores['subjectivity']
            print(f"[TB: {tb_scores['polarity']:.2f}]", end=" ")
        
        if method in ['vader', 'all']:
            vader_scores = analyze_sentiment_vader(text)
            sentiment_df.at[idx, 'vader_compound'] = vader_scores['compound']
            sentiment_df.at[idx, 'vader_positive'] = vader_scores['positive']
            sentiment_df.at[idx, 'vader_negative'] = vader_scores['negative']
            sentiment_df.at[idx, 'vader_neutral'] = vader_scores['neutral']
            print(f"[VADER: {vader_scores['compound']:.2f}]", end=" ")
        
        if method in ['finbert', 'all']:
            finbert_scores = analyze_sentiment_finbert(text)
            sentiment_df.at[idx, 'finbert_positive'] = finbert_scores['positive']
            sentiment_df.at[idx, 'finbert_negative'] = finbert_scores['negative']
            sentiment_df.at[idx, 'finbert_neutral'] = finbert_scores['neutral']
            compound = finbert_scores['positive'] - finbert_scores['negative']
            sentiment_df.at[idx, 'finbert_compound'] = compound
            print(f"[FinBERT: {compound:.2f}]", end=" ")
        
        print("Done")
    

    print("Sentiment Analysis Complete")

    
    return sentiment_df


def create_ensemble_sentiment(sentiment_df):
    """Create ensemble sentiment score"""
    print("\nCreating ensemble sentiment score...")
    
    available_methods = []
    
    if 'textblob_polarity' in sentiment_df.columns:
        available_methods.append('textblob_polarity')
    
    if 'vader_compound' in sentiment_df.columns:
        available_methods.append('vader_compound')
    
    if 'finbert_compound' in sentiment_df.columns:
        available_methods.append('finbert_compound')
    
    if len(available_methods) == 0:
        return sentiment_df
    
    sentiment_df['ensemble_sentiment'] = sentiment_df[available_methods].mean(axis=1)
    
    print(f"  Combined {len(available_methods)} methods")
    
    return sentiment_df


def visualize_sentiment_comparison(sentiment_df, ticker):
    """Create visualizations comparing different sentiment methods"""
    print("\nCreating sentiment comparison charts...")
    
    sentiment_df['trading_date'] = pd.to_datetime(sentiment_df['trading_date'])
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    if 'textblob_polarity' in sentiment_df.columns and 'vader_compound' in sentiment_df.columns:
        axes[0].plot(sentiment_df['trading_date'], sentiment_df['textblob_polarity'], 
                    marker='o', label='TextBlob', linewidth=2)
        axes[0].plot(sentiment_df['trading_date'], sentiment_df['vader_compound'], 
                    marker='s', label='VADER', linewidth=2)
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_title('Sentiment Comparison: TextBlob vs VADER')
        axes[0].set_ylabel('Sentiment Score')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    if 'finbert_compound' in sentiment_df.columns:
        axes[1].plot(sentiment_df['trading_date'], sentiment_df['finbert_compound'], 
                    marker='^', label='FinBERT', color='green', linewidth=2)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_title('Sentiment: FinBERT')
        axes[1].set_ylabel('Sentiment Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    if 'ensemble_sentiment' in sentiment_df.columns:
        axes[2].plot(sentiment_df['trading_date'], sentiment_df['ensemble_sentiment'], 
                    marker='D', label='Ensemble Average', color='purple', linewidth=2.5)
        axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[2].fill_between(sentiment_df['trading_date'], 
                            sentiment_df['ensemble_sentiment'], 0, 
                            alpha=0.3, color='purple')
        axes[2].set_title('Ensemble Sentiment')
        axes[2].set_ylabel('Sentiment Score')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{ticker} - Sentiment Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def show_sentiment_statistics(sentiment_df):
    """Display summary statistics for sentiment scores"""

    print("Sentiment Statistics")

    
    sentiment_cols = [col for col in sentiment_df.columns if 
                     'sentiment' in col.lower() or 'polarity' in col.lower() or 
                     'compound' in col.lower()]
    
    for col in sentiment_cols:
        print(f"{col}:")
        print(f"  Mean: {sentiment_df[col].mean():.3f}")
        print(f"  Min: {sentiment_df[col].min():.3f}")
        print(f"  Max: {sentiment_df[col].max():.3f}\n")


def save_sentiment_data(sentiment_df, ticker):
    """Save sentiment analysis results"""
    filename = f"{ticker}_sentiment_scores.csv"
    sentiment_df.to_csv(filename, index=False)
    print(f" Saved to: {filename}")
    return filename


def run_sentiment_analysis_workflow(ticker, aligned_news, method='all'):
    """Complete workflow for sentiment analysis"""
    print(f"Sentiment Analysis Workflow for {ticker}")

    sentiment_file = f"{ticker}_sentiment_scores.csv"
    
    if os.path.exists(sentiment_file):
        print(f"\nExisting sentiment data found")
        use_existing = input("Use existing data? (y/n): ").lower()
        if use_existing == 'y':
            sentiment_df = pd.read_csv(sentiment_file)
            print(f"Loaded {len(sentiment_df)} days")
            return sentiment_df
    
    sentiment_df = apply_sentiment_analysis(aligned_news, method)
    sentiment_df = create_ensemble_sentiment(sentiment_df)
    save_sentiment_data(sentiment_df, ticker)
    show_sentiment_statistics(sentiment_df)
    visualize_sentiment_comparison(sentiment_df, ticker)
    
    return sentiment_df


# =======================
# SECTION 4: ARIMA MODEL
# =======================

def train_arima_model(prices, future_days=5, order=(1, 1, 1)):
    """Train ARIMA model"""
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
# SECTION 5: SARIMA MODEL
# =======================

def train_sarima_model(prices, future_days=5, order=(2, 1, 1), seasonal_order=(1, 1, 1, 5)):
    """Train SARIMA model"""
    print(f"  Training SARIMA...")
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
# SECTION 6: LSTM MODEL
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
    print(f"  Training LSTM...")
    
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
# SECTION 7: RNN MODEL
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
    print(f"  Training RNN...")
    
    X, y, scaler = prepare_lstm_data(data, look_back_days, future_days)
    
    if len(X) == 0:
        print(f"  Not enough data")
        return None

    model = build_rnn_model(look_back_days, future_days, rnn_units, dropout)
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    last_sequence = X[-1:]
    pred_scaled = model.predict(last_sequence, verbose=0)[0]
    predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    print(f"  RNN trained")
    return predictions


# =======================
# SECTION 8: ENSEMBLE EXPERIMENTATION
# =======================

def run_ensemble_experimentation(data, ticker, future_days=5, look_back_days=60):
    """Test LSTM, RNN, ARIMA, and SARIMA configurations"""
    print(f"Ensemble Experimentation - {ticker}")
    
    results = {}
    prices = data['Close']
    
    # ARIMA
    print("1. Testing ARIMA")
    arima_configs = [
        {'order': (1, 1, 1), 'name': 'ARIMA (1,1,1)'},
        {'order': (2, 1, 1), 'name': 'ARIMA (2,1,1)'},
    ]
    
    for config in arima_configs:
        pred = train_arima_model(prices, future_days, config['order'])
        if pred is not None:
            results[config['name']] = pred
    
    # SARIMA
    print("\n2. Testing SARIMA")
    sarima_configs = [
        {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 5), 'name': 'SARIMA (1,1,1)x(1,1,1,5)'},
    ]
    
    for config in sarima_configs:
        pred = train_sarima_model(prices, future_days, config['order'], config['seasonal_order'])
        if pred is not None:
            results[config['name']] = pred
    
    # LSTM
    print("\n3. Testing LSTM")
    lstm_configs = [
        {'lstm_units': 50, 'dropout': 0.2, 'epochs': 10, 'name': 'LSTM (50 units, 10 epochs)'},
    ]
    
    for config in lstm_configs:
        pred = train_lstm_model(data, future_days, look_back_days, 
                               config['lstm_units'], config['dropout'], config['epochs'])
        if pred is not None:
            results[config['name']] = pred
    
    # RNN
    print("\n4. Testing RNN")
    rnn_configs = [
        {'rnn_units': 50, 'dropout': 0.2, 'epochs': 10, 'name': 'RNN (50 units, 10 epochs)'},
    ]
    
    for config in rnn_configs:
        pred = train_rnn_model(data, future_days, look_back_days, 
                              config['rnn_units'], config['dropout'], config['epochs'])
        if pred is not None:
            results[config['name']] = pred
    
    # Ensemble
    print("\n5. Creating Ensemble")
    
    if len(results) >= 2:
        all_predictions = np.array(list(results.values()))
        ensemble_avg = np.mean(all_predictions, axis=0)
        results['ENSEMBLE_AVERAGE'] = ensemble_avg
        print("Ensemble Average created")
    
    # Display Results
    print("Prediction Results")
    
    for model_name, predictions in results.items():
        print(f"{model_name}:")
        for day in range(future_days):
            print(f"  Day {day+1}: ${predictions[day]:.2f}", end="  ")
        print("\n")
    
    # Plot
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
    
    plt.title(f'{ticker} - Ensemble Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Days Ahead', fontsize=12)
    plt.ylabel('Predicted Stock Price ($)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =======================
# SECTION 9: CLASSIFICATION MODEL (UP/DOWN PREDICTION)
# =======================

def create_classification_target(data):
    """
    Trying to make it binary so if tomorrow's price be higher or lower
    1 = UP (tomorrow > today)
    0 = DOWN (tomorrow <= today)
    """
    print(f"\nCreating classification target...")
    
    data = data.copy()
    data['next_close'] = data['Close'].shift(-1)
    data['target'] = (data['next_close'] > data['Close']).astype(int)
    data = data[:-1]  # Remove last row (no tomorrow data)
    
    up_days = data['target'].sum()
    down_days = len(data) - up_days
    
    print(f"Target created:")
    print(f"   UP days: {up_days} ({up_days/len(data)*100:.1f}%)")
    print(f"   DOWN days: {down_days} ({down_days/len(data)*100:.1f}%)")
    
    return data


def create_technical_indicators(data):
    """
    Calculate MA, RSI, volatility and volume indicators
    
    """
    print(f"\n Creating technical indicators...")
    
    df = data.copy()
    
    # Price changes
    df['price_change'] = df['Close'] - df['Open']
    df['price_change_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # Moving averages (trend indicators)
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['ma_50'] = df['Close'].rolling(window=50).mean()
    
    # MA crossovers (buy/sell signals)
    df['ma_5_10_diff'] = df['ma_5'] - df['ma_10']
    df['ma_10_20_diff'] = df['ma_10'] - df['ma_20']
    
    # Volatility
    df['volatility_5'] = df['Close'].rolling(window=5).std()
    df['volatility_10'] = df['Close'].rolling(window=10).std()
    
    # RSI (momentum)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume indicators
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_5']
    
    # Price range
    df['daily_range'] = df['High'] - df['Low']
    df['daily_range_pct'] = (df['High'] - df['Low']) / df['Open'] * 100
    
    # Close position in range
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    print(f"Created 20+ technical indicators")
    
    return df


def create_lagged_features(data, lags=[1, 2, 3, 5]):
    """
    Create lagged price, volume and sentiment features
    """
    print(f"\n Creating lagged features...")
    
    df = data.copy()
    
    # Lag price and volume
    for lag in lags:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
    
    # Lag sentiment if available
    if 'sentiment' in df.columns:
        for lag in lags:
            df[f'sentiment_lag_{lag}'] = df['sentiment'].shift(lag)
    
    print(f"Created lagged features")
    
    return df


def merge_sentiment_with_stock(stock_data, ticker):
    """
    Merge sentiment scores with stock data
    """
    print(f"\nMerging sentiment with stock data...")
    
    sentiment_file = f"{ticker}_sentiment_scores.csv"
    
    if not os.path.exists(sentiment_file):
        print(f"No sentiment file found. Using stock data only.")
        return stock_data
    
    # Load sentiment
    sentiment_df = pd.read_csv(sentiment_file)
    sentiment_df['trading_date'] = pd.to_datetime(sentiment_df['trading_date']).dt.date
    
    # Merge
    stock_df = stock_data.copy()
    stock_df['date'] = stock_df.index.date
    
    merged = stock_df.merge(
        sentiment_df[['trading_date', 'ensemble_sentiment']],
        left_on='date',
        right_on='trading_date',
        how='left'
    )
    
    # Rename sentiment column
    if 'ensemble_sentiment' in merged.columns:
        merged = merged.rename(columns={'ensemble_sentiment': 'sentiment'})
        merged['sentiment'] = merged['sentiment'].fillna(0)
        print(f"Sentiment merged: {(merged['sentiment'] != 0).sum()} days")
    
    # Drop extra columns
    merged = merged.drop(['trading_date', 'date'], axis=1, errors='ignore')
    merged = merged.set_index(stock_data.index)
    
    return merged


def prepare_classification_features(data):
    """
    Select features for classification model
    
    FEATURE SELECTION:
    1. Price features: Open, High, Low, Close, Volume
    2. Technical indicators: MA, RSI, Volatility
    3. Sentiment: News sentiment score (if available)
    4. Lagged values: Past days' data
    """
    print(f"\nSelecting features for classification...")
    
    feature_cols = []
    
    # Basic price features
    basic = ['Open', 'High', 'Low', 'Close', 'Volume',
             'price_change', 'price_change_pct', 'daily_range', 'daily_range_pct']
    feature_cols.extend(basic)
    
    # Technical indicators
    technical = ['ma_5', 'ma_10', 'ma_20', 'ma_50',
                'ma_5_10_diff', 'ma_10_20_diff',
                'volatility_5', 'volatility_10', 'rsi']
    feature_cols.extend(technical)
    
    # Volume features
    volume = ['volume_change', 'volume_ratio']
    feature_cols.extend(volume)
    
    # Position features
    position = ['close_position']
    feature_cols.extend(position)
    
    # Sentiment (if available)
    if 'sentiment' in data.columns:
        feature_cols.append('sentiment')
        print(f"   Using sentiment feature")
    
    # Lagged features
    lagged = [col for col in data.columns if 'lag' in col]
    feature_cols.extend(lagged)
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in data.columns]
    
    X = data[feature_cols]
    y = data['target']
    
    print(f"Selected {len(feature_cols)} features:")
    print(f"    Price: {len(basic)}")
    print(f"    Technical: {len(technical)}")
    print(f"    Volume: {len(volume)}")
    print(f"    Sentiment: {1 if 'sentiment' in feature_cols else 0}")
    print(f"    Lagged: {len(lagged)}")
    
    return X, y, feature_cols


def train_classification_models(X_train, y_train):
    """
    Training 3 classification models
    """

    print(" Training Classification Models ")
    
    models = {}
    
    # Logistic Regression
    print("Training Logistic Regression...", end=" ")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    print("Done")
    
    # Random Forest
    print("Training Random Forest...", end=" ")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    print("Done")
    
    # Gradient Boosting
    print("Training Gradient Boosting...", end=" ")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    print("Done")
    
    return models


def evaluate_classification_models(models, X_train, y_train, X_test, y_test):
    """Evaluate all classification models"""
    print(" Model Evaluation ")
    
    results = {}
    
    for name, model in models.items():
        print(f" {name} ")
        
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n Accuracy:")
        print(f"   Training:  {train_acc*100:.2f}%")
        print(f"   Testing:   {test_acc*100:.2f}%")
        
        print(f"\n Detailed Metrics:")
        print(classification_report(y_test, test_pred, 
                                   target_names=['DOWN (0)', 'UP (1)']))
        
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'predictions': test_pred
        }
    
    return results


def plot_classification_results(results, y_test, feature_cols, ticker):
    """Plot confusion matrices and feature importance"""
    print("\n Creating visualisations...")
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{name}\nAccuracy: {result["test_acc"]*100:.2f}%', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_xticklabels(['DOWN', 'UP'])
        axes[idx].set_yticklabels(['DOWN', 'UP'])
    
    plt.suptitle(f'{ticker} - Classification Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{ticker}_classification_confusion.png', dpi=300)
    print(f"Saved: {ticker}_classification_confusion.png")
    plt.show()
    
    # Feature importance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    rf = results['Random Forest']['model']
    rf_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    axes[0].barh(rf_importance['feature'], rf_importance['importance'])
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Random Forest - Top 15 Features', fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    gb = results['Gradient Boosting']['model']
    gb_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': gb.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    axes[1].barh(gb_importance['feature'], gb_importance['importance'])
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Gradient Boosting - Top 15 Features', fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'{ticker} - Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{ticker}_feature_importance.png', dpi=300)
    print(f"Saved: {ticker}_feature_importance.png")
    plt.show()


def run_classification_workflow(data, ticker):
    """
    Complete workflow for UP/DOWN classification
    """
    print(f" Classification Workflow: Up/Down ")
    
    # Merge sentiment
    data = merge_sentiment_with_stock(data, ticker)
    
    # Create target
    data = create_classification_target(data)
    
    # Create features
    data = create_technical_indicators(data)
    data = create_lagged_features(data)
    
    # Remove NaN
    original_len = len(data)
    data = data.dropna()
    print(f"\n Removed {original_len - len(data)} rows with NaN")
    print(f"Final dataset: {len(data)} days")
    
    # Prepare features
    X, y, feature_cols = prepare_classification_features(data)
    
    # Train/test split (80/20, keeping time order)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f" Train/Test Split:")
    print(f"   Training: {len(X_train)} days")
    print(f"   Testing: {len(X_test)} days")
    
    # Scale features
    print(f"\n Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f" Features scaled")
    
    # Train models
    models = train_classification_models(X_train_scaled, y_train)
    
    # Evaluate
    results = evaluate_classification_models(models, X_train_scaled, y_train,
                                            X_test_scaled, y_test)
    
    # Visualize
    plot_classification_results(results, y_test, feature_cols, ticker)
    
    # Summary
    print(f" Classification Complete ")
    
    print(f"\n Model Comparison")
    for name, result in results.items():
        print(f"   {name:25s} {result['test_acc']*100:.2f}%")
    
    best_name = max(results.items(), key=lambda x: x[1]['test_acc'])[0]
    best_acc = results[best_name]['test_acc']
    print(f"\n Best Model: {best_name} ({best_acc*100:.2f}%)")
    
    print(f"\n Interpretation:")
    print(f"   If accuracy > 50%: Model can predict better than random")
    print(f"   If accuracy > 55%: Model shows good predictive power")
    print(f"   If accuracy > 60%: Model shows strong predictive power")

# =======================
# SECTION 10: BASELINE COMPARISON (WITH vs WITHOUT SENTIMENT)
# =======================

def compare_with_without_sentiment(data, ticker):
    """
    Compare model performance WITH sentiment vs WITHOUT sentiment
    This answers: "Does sentiment data actually help?"
    """
    print(f" Baseline Comparison: WITH vs WITHOUT SENTIMENT ")
    print("\nWe will train models TWICE:")
    print("  1. WITHOUT sentiment (baseline)")
    print("  2. WITH sentiment (full model)")
    print("\nThen compare which performs better!")
    
    # ============================================
    # PREPARE DATA WITHOUT SENTIMENT
    # ============================================
    print(" Baseline Model (no sentiment)")
    
    # Don't merge sentiment - use stock data only
    data_no_sent = data.copy()
    data_no_sent = create_classification_target(data_no_sent)
    data_no_sent = create_technical_indicators(data_no_sent)
    data_no_sent = create_lagged_features(data_no_sent)
    data_no_sent = data_no_sent.dropna()
    
    # Prepare features WITHOUT sentiment
    X_no_sent, y_no_sent, features_no_sent = prepare_classification_features(data_no_sent)
    
    # Train/test split
    split_idx = int(len(X_no_sent) * 0.8)
    X_train_no, X_test_no = X_no_sent[:split_idx], X_no_sent[split_idx:]
    y_train_no, y_test_no = y_no_sent[:split_idx], y_no_sent[split_idx:]
    
    print(f"\n Baseline Dataset:")
    print(f"   Training: {len(X_train_no)} days")
    print(f"   Testing: {len(X_test_no)} days")
    print(f"   Features: {len(features_no_sent)} (No sentiment)")
    
    # Scale
    scaler_no = StandardScaler()
    X_train_no_scaled = scaler_no.fit_transform(X_train_no)
    X_test_no_scaled = scaler_no.transform(X_test_no)
    
    # Train models
    models_no_sent = train_classification_models(X_train_no_scaled, y_train_no)
    results_no_sent = evaluate_classification_models(
        models_no_sent, X_train_no_scaled, y_train_no, 
        X_test_no_scaled, y_test_no
    )
    
    # ============================================
    # PREPARE DATA WITH SENTIMENT
    # ============================================
    print(" Full Model (With Sentiment) ")

    
    # Merge sentiment
    data_with_sent = merge_sentiment_with_stock(data, ticker)
    data_with_sent = create_classification_target(data_with_sent)
    data_with_sent = create_technical_indicators(data_with_sent)
    data_with_sent = create_lagged_features(data_with_sent)
    data_with_sent = data_with_sent.dropna()
    
    # Prepare features with sentiment
    X_with_sent, y_with_sent, features_with_sent = prepare_classification_features(data_with_sent)
    
    # Train/test split
    split_idx = int(len(X_with_sent) * 0.8)
    X_train_with, X_test_with = X_with_sent[:split_idx], X_with_sent[split_idx:]
    y_train_with, y_test_with = y_with_sent[:split_idx], y_with_sent[split_idx:]
    
    print(f"\n Full Model Dataset:")
    print(f"   Training: {len(X_train_with)} days")
    print(f"   Testing: {len(X_test_with)} days")
    print(f"   Features: {len(features_with_sent)} (WITH sentiment)")
    
    # Scale
    scaler_with = StandardScaler()
    X_train_with_scaled = scaler_with.fit_transform(X_train_with)
    X_test_with_scaled = scaler_with.transform(X_test_with)
    
    # Train models
    models_with_sent = train_classification_models(X_train_with_scaled, y_train_with)
    results_with_sent = evaluate_classification_models(
        models_with_sent, X_train_with_scaled, y_train_with,
        X_test_with_scaled, y_test_with
    )
    
    # ============================================
    # SIDE-BY-SIDE COMPARISON
    # ============================================
    print("  Comparison Results ")
    
    # Create comparison table
    comparison_data = []
    
    for model_name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:
        no_sent_acc = results_no_sent[model_name]['test_acc']
        with_sent_acc = results_with_sent[model_name]['test_acc']
        improvement = (with_sent_acc - no_sent_acc) * 100
        
        comparison_data.append({
            'Model': model_name,
            'Without Sentiment': f"{no_sent_acc*100:.2f}%",
            'With Sentiment': f"{with_sent_acc*100:.2f}%",
            'Improvement': f"{improvement:+.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print(comparison_df.to_string(index=False))

    # Calculate metrics for best model (Logistic Regression)

    print(" Detailed Metrics (LOGISTIC REGRESSION) ")
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # WITHOUT sentiment
    lr_no = results_no_sent['Logistic Regression']
    y_pred_no = lr_no['predictions']
    
    print("\n WITHOUT SENTIMENT (Baseline):")
    print(f"   Accuracy:  {lr_no['test_acc']*100:.2f}%")
    print(f"   Precision: {precision_score(y_test_no, y_pred_no)*100:.2f}%")
    print(f"   Recall:    {recall_score(y_test_no, y_pred_no)*100:.2f}%")
    print(f"   F1-Score:  {f1_score(y_test_no, y_pred_no)*100:.2f}%")
    
    # WITH sentiment
    lr_with = results_with_sent['Logistic Regression']
    y_pred_with = lr_with['predictions']
    
    print("\n WITH SENTIMENT (Full Model):")
    print(f"   Accuracy:  {lr_with['test_acc']*100:.2f}%")
    print(f"   Precision: {precision_score(y_test_with, y_pred_with)*100:.2f}%")
    print(f"   Recall:    {recall_score(y_test_with, y_pred_with)*100:.2f}%")
    print(f"   F1-Score:  {f1_score(y_test_with, y_pred_with)*100:.2f}%")
    
    # ============================================
    # VISUALIZATIONS
    # ============================================
    
    # 1. Side-by-side confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Without sentiment
    cm_no = confusion_matrix(y_test_no, y_pred_no)
    sns.heatmap(cm_no, annot=True, fmt='d', cmap='Reds', ax=axes[0], cbar=False)
    axes[0].set_title(f'WITHOUT Sentiment\nAccuracy: {lr_no["test_acc"]*100:.2f}%', 
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    axes[0].set_xticklabels(['DOWN', 'UP'])
    axes[0].set_yticklabels(['DOWN', 'UP'])
    
    # With sentiment
    cm_with = confusion_matrix(y_test_with, y_pred_with)
    sns.heatmap(cm_with, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
    axes[1].set_title(f'WITH Sentiment\nAccuracy: {lr_with["test_acc"]*100:.2f}%', 
                      fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')
    axes[1].set_xticklabels(['DOWN', 'UP'])
    axes[1].set_yticklabels(['DOWN', 'UP'])
    
    plt.suptitle(f'{ticker} - Baseline Comparison (Logistic Regression)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{ticker}_baseline_comparison.png', dpi=300)
    print(f"\n Saved: {ticker}_baseline_comparison.png")
    plt.show()
    
    # 2. Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting']
    no_sent_scores = [results_no_sent[m]['test_acc']*100 for m in 
                      ['Logistic Regression', 'Random Forest', 'Gradient Boosting']]
    with_sent_scores = [results_with_sent[m]['test_acc']*100 for m in 
                        ['Logistic Regression', 'Random Forest', 'Gradient Boosting']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, no_sent_scores, width, label='Without Sentiment', 
                   color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, with_sent_scores, width, label='With Sentiment', 
                   color='seagreen', alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{ticker} - Impact of Sentiment on Model Performance', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, 
               label='Random Chance')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_sentiment_impact.png', dpi=300)
    print(f" Saved: {ticker}_sentiment_impact.png")
    plt.show()
    
    # ============================================
    # FINAL VERDICT
    # ============================================
    print(" Final Verdict ")
    
    avg_no_sent = np.mean([r['test_acc'] for r in results_no_sent.values()])
    avg_with_sent = np.mean([r['test_acc'] for r in results_with_sent.values()])
    
    print(f"\nAverage Accuracy:")
    print(f"   WITHOUT Sentiment: {avg_no_sent*100:.2f}%")
    print(f"   WITH Sentiment:    {avg_with_sent*100:.2f}%")
    print(f"   Improvement:       {(avg_with_sent - avg_no_sent)*100:+.2f}%")
    
    if avg_with_sent > avg_no_sent:
        print(f"\n Sentiment Data is Valuable")
        print(f"   Models perform better when sentiment is included.")
    elif avg_with_sent < avg_no_sent:
        print(f"\n Sentiment Data was not helpful")
        print(f"   Models perform better without sentiment.")
        print(f"   Possible reasons:")
        print(f"   - Limited news data (only 30 days)")
        print(f"   - News may lag actual price movements")
        print(f"   - Technical indicators are stronger predictors")
    else:
        print(f"\n No Significant Difference")
        print(f"   Sentiment had minimal impact.")
    
    # Save comparison results
    comparison_df.to_csv(f'{ticker}_baseline_comparison.csv', index=False)
    print(f"\n Saved: {ticker}_baseline_comparison.csv")
    
    return {
        'without_sentiment': results_no_sent,
        'with_sentiment': results_with_sent,
        'comparison': comparison_df
    }
# =======================
# MAIN MENU (UPDATED)
# =======================

print(" Stock Prediction Tool v0.6 ")  

stock = input("\nEnter stock symbol (like AAPL, TSLA, GOOGL): ").upper() or "AAPL"
data = get_stock_data(stock)

if data is not None:
    print("\nWhat do you want to do?")
    print("1 = Make charts (candlestick + boxplot)")
    print("2 = Run ensemble (LSTM + RNN + ARIMA + SARIMA)")
    print("3 = Collect news data")
    print("4 = Run sentiment analysis")
    print("5 = Complete sentiment workflow (news + sentiment)")
    print("6 = Classification Model (Predict UP/DOWN) ")
    print("7 = Baseline Comparison (With vs Without Sentiment) ")  
    
    main_choice = input("\nEnter 1-7: ")  

    if main_choice == "1":
        make_candlestick_chart(data, stock, days_per_candle=1)
        make_boxplot_chart(data, stock, window_size=5)

    elif main_choice == "2":
        future_days = int(input("Days to predict (default=5): ") or "5")
        look_back = int(input("Past days to use (default=60): ") or "60")
        run_ensemble_experimentation(data, stock, future_days, look_back)

    elif main_choice == "3":
        print("\nGet FREE NewsAPI key: https://newsapi.org/")
        api_key = input("Enter your NewsAPI key: ").strip()
        
        if not api_key:
            print("API key required!")
        else:
            news_data = collect_all_news(stock, data, api_key)
            if news_data is not None:
                print(f"\n Files created: {stock}_news_raw.csv, {stock}_news_aligned.csv")

    elif main_choice == "4":
        aligned_file = f"{stock}_news_aligned.csv"
        
        if not os.path.exists(aligned_file):
            print(f"\n No aligned news found! Run option 3 first.")
        else:
            aligned_news = pd.read_csv(aligned_file)
            
            print("\nWhich sentiment method?")
            print("1 = TextBlob only")
            print("2 = VADER only")
            print("3 = FinBERT only")
            print("4 = All methods (recommended)")
            
            method_choice = input("Enter 1-4: ")
            method_map = {'1': 'textblob', '2': 'vader', '3': 'finbert', '4': 'all'}
            method = method_map.get(method_choice, 'all')
            
            sentiment_data = run_sentiment_analysis_workflow(stock, aligned_news, method)

    elif main_choice == "5":
        print("\nGet FREE NewsAPI key: https://newsapi.org/")
        api_key = input("Enter your NewsAPI key: ").strip()
        
        if not api_key:
            print("API key required!")
        else:
            # Collect news
            news_data = collect_all_news(stock, data, api_key)
            
            if news_data is not None:
                # Run sentiment analysis
                sentiment_data = run_sentiment_analysis_workflow(stock, news_data, method='all')

    elif main_choice == "6":
        print("\n Starting Classification Model...")
        print("\nThis will:")
        print("  1. Create technical indicators (MA, RSI, etc.)")
        print("  2. Merge with sentiment data (if available)")
        print("  3. Train 3 models: Logistic, Random Forest, Gradient Boosting")
        print("  4. Predict: Will tomorrow's price go UP or DOWN?")
        
        proceed = input("\nProceed? (y/n): ").lower()
        
        if proceed == 'y':
            run_classification_workflow(data, stock)
        else:
            print("Cancelled.")

    elif main_choice == "7":  
        print("\n Starting Baseline Comparison...")
        print("\nThis will train models TWICE and compare:")
        print("  1. WITHOUT sentiment (baseline)")
        print("  2. WITH sentiment (full model)")
        print("  Then show which performs better!")
        
        proceed = input("\nProceed? (y/n): ").lower()
        
        if proceed == 'y':
            comparison_results = compare_with_without_sentiment(data, stock)
        else:
            print("Cancelled.")

    else:
        print("Invalid choice!")

else:
    print("Failed to get stock data!")

print(" Done ")