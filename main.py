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

# Real-time Data Streaming
def get_real_time_data(symbol):
    """Fetch real-time data from Binance"""
    ticker = BINANCE_API.fetch_ticker(f"{symbol.upper()}/USDT")
    return {
        'price': ticker['last'],
        'volume': ticker['baseVolume'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Sentiment Analysis
def get_sentiment(symbol):
    """Fetch social media sentiment from Twitter"""
    query = f"{symbol} cryptocurrency"
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
    """Calculate historical volatility"""
    returns = data['prices'].pct_change().dropna()
    return returns.std() * np.sqrt(365)  # Annualized volatility

# Risk/Reward Calculation
def calculate_risk_reward(data, prediction):
    """Calculate risk/reward ratio"""
    current_price = data['prices'].iloc[-1]
    stop_loss = current_price * 0.95  # 5% stop loss
    take_profit = prediction * 1.10  # 10% target
    risk = current_price - stop_loss
    # Avoid division by zero
    if risk == 0:
        risk = 1e-6
    reward = take_profit - current_price
    return reward / risk

# Enhanced Prediction Strategies
def technical_analysis_prediction(data):
    """Strategy 1: Technical Indicators"""
    df = data.copy()
    # Calculate moving averages and RSI
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
    confidence = (len(signals) / 4) * (1 - volatility)  # Adjust confidence for volatility
    
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
    # Ensure we have enough data points (at least 60)
    if len(data) < 60:
        raise ValueError("Not enough data for LSTM prediction.")
        
    scaled_data = scaler.fit_transform(data['prices'].values.reshape(-1, 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    # Training the model on the available data
    model.fit(scaled_data, scaled_data, epochs=10, batch_size=32, verbose=0)
    
    future_preds = []
    current_batch = scaled_data[-60:].reshape(1, 60, 1)
    for _ in range(DAYS_TO_PREDICT):
        next_pred = model.predict(current_batch)
        # next_pred shape: (1, 1)
        future_preds.append(next_pred[0, 0])
        # Reshape next_pred to (1,1,1) before concatenation
        next_pred_reshaped = next_pred.reshape(1, 1, 1)
        current_batch = np.concatenate([current_batch[:, 1:, :], next_pred_reshaped], axis=1)
        
    prediction = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))[ -1, 0]
    volatility = calculate_volatility(data)
    confidence = 0.7 * (1 - volatility)  # Adjust confidence for volatility
    
    return {
        'strategy': 'LSTM',
        'prediction': prediction,
        'confidence': confidence,
        'volatility': volatility
    }

# Data Fetching
def get_crypto_data(symbol):
    url = f"{COINGECKO_API}/coins/{symbol}/market_chart?vs_currency=usd&days=365"
    response = requests.get(url)
    data = response.json()
    if 'prices' not in data or 'total_volumes' not in data:
        raise ValueError("Invalid data received from CoinGecko API.")
    
    # Create DataFrame for prices and volumes
    prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'prices'])
    volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    
    # Merge the dataframes on the timestamp column
    df = pd.merge(prices_df, volumes_df, on='timestamp')
    # Convert timestamp (which is in ms) to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def prophet_prediction(data):
    """Strategy 3: Facebook's Prophet"""
    df = data.rename(columns={'timestamp': 'ds', 'prices': 'y'})
    # Prophet requires ds to be datetime
    df['ds'] = pd.to_datetime(df['ds'])
    # Remove any potential NaN values
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("Not enough data for Prophet model.")
        
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=DAYS_TO_PREDICT)
    forecast = model.predict(future)
    
    volatility = calculate_volatility(data)
    # Compute confidence based on the forecast range and volatility
    conf_range = forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]
    confidence = (1 - (conf_range / forecast['yhat'].iloc[-1])) * (1 - volatility)
    
    return {
        'strategy': 'Prophet',
        'prediction': forecast['yhat'].iloc[-1],
        'confidence': confidence,
        'volatility': volatility
    }

# Telegram Bot Handlers
# (Optional: if you wish to use this handler, update its signature.)
async def analyze_crypto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].lower() if context.args else 'BTC'
    
    try:
        data = get_crypto_data(symbol)
        real_time_data = get_real_time_data(symbol)
        sentiment = get_sentiment(symbol)
        
        results = []
        results.append(technical_analysis_prediction(data))
        results.append(lstm_prediction(data))
        results.append(prophet_prediction(data))
        
        best_strategy = max(results, key=lambda x: x['confidence'])
        risk_reward = calculate_risk_reward(data, best_strategy['prediction'])
        
        # Format Response
        response = (
            f"📈 **{symbol.upper()} {DAYS_TO_PREDICT}-Day Predictions**\n\n"
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
    await update.message.reply_text("🚀 Crypto Prediction Bot\nUse /predict [coin_id]")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].lower() if context.args else 'bitcoin'
    try:
        data = get_crypto_data(symbol)
        real_time_data = get_real_time_data(symbol)
        sentiment = get_sentiment(symbol)
        
        results = []
        results.append(technical_analysis_prediction(data))
        results.append(lstm_prediction(data))
        results.append(prophet_prediction(data))
        
        best_strategy = max(results, key=lambda x: x['confidence'])
        risk_reward = calculate_risk_reward(data, best_strategy['prediction'])
        
        response = (
            f"📈 **{symbol.upper()} {DAYS_TO_PREDICT}-Day Predictions**\n\n"
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
    
    # If you wish to use analyze_crypto as an additional command, uncomment the next line:
    # application.add_handler(CommandHandler("analyze", analyze_crypto))
    
    application.run_polling()

if __name__ == '__main__':
    main()
