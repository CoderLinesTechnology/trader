import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
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

# Configuration
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
COINGECKO_API = "https://api.coingecko.com/api/v3"
BINANCE_API = ccxt.binance()
DAYS_TO_PREDICT = 7

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
    """Fetch social media sentiment (e.g., Twitter)"""
    query = f"{symbol} cryptocurrency"
    response = requests.get(SENTIMENT_API, params={'query': query, 'max_results': 100})
    tweets = response.json().get('data', [])
    sentiments = [TextBlob(tweet['text']).sentiment.polarity for tweet in tweets]
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
    confidence = len(signals) / 4 * (1 - volatility)  # Adjust confidence for volatility
    
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
    scaled_data = scaler.fit_transform(data['prices'].values.reshape(-1,1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60,1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(scaled_data, scaled_data, epochs=10, batch_size=32, verbose=0)
    
    future_preds = []
    current_batch = scaled_data[-60:].reshape(1,60,1)
    for _ in range(DAYS_TO_PREDICT):
        next_pred = model.predict(current_batch)[0]
        future_preds.append(next_pred[0])
        current_batch = np.append(current_batch[:,1:,:], [[next_pred]], axis=1)
        
    prediction = scaler.inverse_transform([future_preds])[0][-1]
    volatility = calculate_volatility(data)
    confidence = 0.7 * (1 - volatility)  # Adjust confidence for volatility
    
    return {
        'strategy': 'LSTM',
        'prediction': prediction,
        'confidence': confidence,
        'volatility': volatility
    }

def prophet_prediction(data):
    """Strategy 3: Facebook's Prophet"""
    df = data.rename(columns={'timestamp': 'ds', 'prices': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=DAYS_TO_PREDICT)
    forecast = model.predict(future)
    
    volatility = calculate_volatility(data)
    confidence = (1 - (forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / forecast['yhat'].iloc[-1])
    confidence *= (1 - volatility)  # Adjust confidence for volatility
    
    return {
        'strategy': 'Prophet',
        'prediction': forecast['yhat'].iloc[-1],
        'confidence': confidence,
        'volatility': volatility
    }

# Telegram Bot Handlers
def analyze_crypto(update: Update, context: CallbackContext):
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
        
        # Risk/Reward Calculation
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
        
        update.message.reply_text(response)
        
    except Exception as e:
        update.message.reply_text(f"❌ Error: {str(e)}")


# Initialize logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot Handlers
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
        logger.error(f"Error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

# Main Function
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", predict))
    
    application.run_polling()

if __name__ == '__main__':
    main()