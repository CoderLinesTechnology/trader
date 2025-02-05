import os
import logging
import requests
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import ta
from textblob import TextBlob
import ccxt
from datetime import datetime

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Configuration
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TWITTER_BEARER = os.environ.get('TWITTER_BEARER')
COINGECKO_API = "https://api.coingecko.com/api/v3"
BINANCE_API = ccxt.binance()
DAYS_TO_PREDICT = 7
SENTIMENT_API = "https://api.twitter.com/2/tweets/search/recent"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Helper function to fetch the CoinGecko coin id for a given ticker
def get_coin_id(ticker):
    """
    Fetch the coin id from CoinGecko using the /coins/list endpoint.
    For example, if ticker is 'btc', this should return 'bitcoin'.
    """
    url = f"{COINGECKO_API}/coins/list"
    response = requests.get(url)
    response.raise_for_status()
    coins = response.json()
    ticker = ticker.lower()
    for coin in coins:
        if coin.get("symbol", "").lower() == ticker:
            return coin.get("id")
    # Fallback: if no match found, return the ticker itself
    return ticker

# Real-time Data Streaming from Binance
def get_real_time_data(ticker):
    """
    Fetch real-time data from Binance using the ticker symbol.
    For Binance, the ticker should be in uppercase (e.g. "BTC" becomes "BTC/USDT").
    """
    ticker_data = BINANCE_API.fetch_ticker(f"{ticker.upper()}/USDT")
    return {
        'price': ticker_data['last'],
        'volume': ticker_data['baseVolume'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Sentiment Analysis from Twitter
def get_sentiment(ticker):
    """
    Fetch social media sentiment from Twitter.
    Uses the ticker in the query (e.g. "btc cryptocurrency").
    """
    query = f"{ticker} cryptocurrency"
    params = {'query': query, 'max_results': 100}
    headers = {"Authorization": f"Bearer {TWITTER_BEARER}"}
    response = requests.get(SENTIMENT_API, params=params, headers=headers)
    tweets = response.json().get('data', [])
    sentiments = [TextBlob(tweet.get('text', '')).sentiment.polarity for tweet in tweets]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    return {
        'sentiment': avg_sentiment,
        'tweet_count': len(tweets)
    }

# Volatility Calculation
def calculate_volatility(data):
    """
    Calculate annualized historical volatility from price data.
    """
    returns = data['prices'].pct_change().dropna()
    return returns.std() * np.sqrt(365)

# Risk/Reward Calculation
def calculate_risk_reward(data, prediction):
    """
    Calculate a risk/reward ratio based on the current price and prediction.
    """
    current_price = data['prices'].iloc[-1]
    stop_loss = current_price * 0.95  # 5% stop loss
    take_profit = prediction * 1.10  # 10% target
    risk = current_price - stop_loss
    if risk == 0:
        risk = 1e-6  # Avoid division by zero
    reward = take_profit - current_price
    return reward / risk

# Enhanced Prediction Strategies
def technical_analysis_prediction(data):
    """Strategy 1: Technical Indicators"""
    df = data.copy()
    df['ma50'] = df['prices'].rolling(50).mean()
    df['ma200'] = df['prices'].rolling(200).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['prices'], window=14).rsi()
    df['volume_ma'] = df['volume'].rolling(14).mean()
    
    signals = []
    if df['ma50'].iloc[-1] > df['ma200'].iloc[-1]:
        signals.append("Bullish MA Crossover")
    if df['rsi'].iloc[-1] < 30:
        signals.append("Oversold RSI")
    elif df['rsi'].iloc[-1] > 70:
        signals.append("Overbought RSI")
    if df['volume'].iloc[-1] > df['volume_ma'].iloc[-1]:
        signals.append("High Volume")
        
    prediction = df['prices'].iloc[-1] * (1 + (0.001 * len(signals)))
    volatility = calculate_volatility(df)
    confidence = (len(signals) / 4) * (1 - volatility)
    
    return {
        'strategy': 'Technical Analysis',
        'prediction': prediction,
        'confidence': confidence,
        'signals': signals,
        'volatility': volatility
    }

def lstm_prediction(data):
    """Strategy 2: LSTM Neural Network"""
    scaler = MinMaxScaler()
    if len(data) < 60:
        raise ValueError("Not enough data for LSTM prediction.")
        
    scaled_data = scaler.fit_transform(data['prices'].values.reshape(-1, 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(scaled_data, scaled_data, epochs=10, batch_size=32, verbose=0)
    
    future_preds = []
    current_batch = scaled_data[-60:].reshape(1, 60, 1)
    for _ in range(DAYS_TO_PREDICT):
        next_pred = model.predict(current_batch)
        future_preds.append(next_pred[0, 0])
        next_pred_reshaped = next_pred.reshape(1, 1, 1)
        current_batch = np.concatenate([current_batch[:, 1:, :], next_pred_reshaped], axis=1)
        
    prediction = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))[-1, 0]
    volatility = calculate_volatility(data)
    confidence = 0.7 * (1 - volatility)
    
    return {
        'strategy': 'LSTM',
        'prediction': prediction,
        'confidence': confidence,
        'volatility': volatility
    }

def prophet_prediction(data):
    """Strategy 3: Facebook's Prophet"""
    df = data.rename(columns={'timestamp': 'ds', 'prices': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("Not enough data for Prophet model.")
        
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=DAYS_TO_PREDICT)
    forecast = model.predict(future)
    
    volatility = calculate_volatility(data)
    conf_range = forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]
    confidence = (1 - (conf_range / forecast['yhat'].iloc[-1])) * (1 - volatility)
    
    return {
        'strategy': 'Prophet',
        'prediction': forecast['yhat'].iloc[-1],
        'confidence': confidence,
        'volatility': volatility
    }

# Data Fetching from CoinGecko using coin id mapping
def get_crypto_data(ticker):
    """
    Fetch historical market data from CoinGecko.
    Converts the given ticker (e.g. "btc") to the appropriate CoinGecko coin id.
    """
    coin_id = get_coin_id(ticker)
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart?vs_currency=usd&days=365"
    response = requests.get(url)
    data = response.json()
    if 'prices' not in data or 'total_volumes' not in data:
        raise ValueError("Invalid data received from CoinGecko API.")
    
    prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'prices'])
    volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    
    df = pd.merge(prices_df, volumes_df, on='timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Telegram Bot Handlers
async def analyze_crypto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ticker = context.args[0].lower() if context.args else 'btc'
    try:
        # For CoinGecko: map ticker to coin id (e.g., "btc" -> "bitcoin")
        data = get_crypto_data(ticker)
        # For Binance: use ticker directly (e.g., "btc" becomes "BTC/USDT")
        real_time_data = get_real_time_data(ticker)
        sentiment = get_sentiment(ticker)
        
        results = [
            technical_analysis_prediction(data),
            lstm_prediction(data),
            prophet_prediction(data)
        ]
        
        best_strategy = max(results, key=lambda x: x['confidence'])
        risk_reward = calculate_risk_reward(data, best_strategy['prediction'])
        
        response = (
            f"📈 **{ticker.upper()}/USDT {DAYS_TO_PREDICT}-Day Predictions**\n\n"
            f"• Real-Time Price: ${real_time_data['price']:.2f}\n"
            f"• Volume (24h): {real_time_data['volume']:.2f}\n"
            f"• Sentiment: {sentiment['sentiment']:.2f} ({sentiment['tweet_count']} tweets)\n\n"
        )
        
        for result in results:
            response += (
                f"• {result['strategy']}:\n"
                f"  Price: ${result['prediction']:.2f}\n"
                f"  Confidence: {result['confidence']*100:.1f}%\n"
                f"  Volatility: {result['volatility']*100:.1f}%\n"
            )
            if 'signals' in result:
                response += f"  Signals: {', '.join(result['signals'])}\n"
        
        response += (
            f"\n🌟 **Recommended Strategy**: {best_strategy['strategy']} "
            f"(Confidence: {best_strategy['confidence']*100:.1f}%)\n"
            f"📊 **Risk/Reward Ratio**: {risk_reward:.2f}\n"
        )
        
        await update.message.reply_text(response)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Crypto Prediction Bot\nUse /predict [ticker]")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ticker = context.args[0].lower() if context.args else 'btc'
    try:
        data = get_crypto_data(ticker)
        real_time_data = get_real_time_data(ticker)
        sentiment = get_sentiment(ticker)
        
        results = [
            technical_analysis_prediction(data),
            lstm_prediction(data),
            prophet_prediction(data)
        ]
        
        best_strategy = max(results, key=lambda x: x['confidence'])
        risk_reward = calculate_risk_reward(data, best_strategy['prediction'])
        
        response = (
            f"📈 **{ticker.upper()}/USDT {DAYS_TO_PREDICT}-Day Predictions**\n\n"
            f"• Real-Time Price: ${real_time_data['price']:.2f}\n"
            f"• Volume (24h): {real_time_data['volume']:.2f}\n"
            f"• Sentiment: {sentiment['sentiment']:.2f} ({sentiment['tweet_count']} tweets)\n\n"
        )
        
        for result in results:
            response += (
                f"• {result['strategy']}:\n"
                f"  Price: ${result['prediction']:.2f}\n"
                f"  Confidence: {result['confidence']*100:.1f}%\n"
                f"  Volatility: {result['volatility']*100:.1f}%\n"
            )
            if 'signals' in result:
                response += f"  Signals: {', '.join(result['signals'])}\n"
        
        response += (
            f"\n🌟 **Recommended Strategy**: {best_strategy['strategy']} "
            f"(Confidence: {best_strategy['confidence']*100:.1f}%)\n"
            f"📊 **Risk/Reward Ratio**: {risk_reward:.2f}\n"
        )
        
        await update.message.reply_text(response)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", predict))
    
    # Optionally, if you wish to use analyze_crypto as an additional command:
    # application.add_handler(CommandHandler("analyze", analyze_crypto))
    
    application.run_polling()

if __name__ == '__main__':
    main()
