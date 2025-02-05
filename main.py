import os
import logging
import asyncio
import aiohttp
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
from time import time
from quart import Quart, jsonify

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import nest_asyncio
nest_asyncio.apply()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Validate required environment variables
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TWITTER_BEARER = os.environ.get('TWITTER_BEARER')
if not TELEGRAM_TOKEN or not TWITTER_BEARER:
    logging.error("Missing required environment variables: TELEGRAM_TOKEN and/or TWITTER_BEARER")
    exit(1)

# Configuration constants
COINGECKO_API = "https://api.coingecko.com/api/v3"
BINANCE_API = ccxt.binance()  # synchronous; we will wrap calls
DAYS_TO_PREDICT = 7
SENTIMENT_API = "https://api.twitter.com/2/tweets/search/recent"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cache = {}


# Static mapping for 20 common cryptocurrencies
COIN_MAPPING = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "doge": "dogecoin",
    "ltc": "litecoin",
    "bnb": "binancecoin",
    "ada": "cardano",
    "xrp": "ripple",
    "dot": "polkadot",
    "link": "chainlink",
    "uni": "uniswap",
    "sol": "solana",
    "avax": "avalanche-2",
    "matic": "matic-network",
    "algo": "algorand",
    "vet": "vechain",
    "fil": "filecoin",
    "atom": "cosmos",
    "trx": "tron",
    "xlm": "stellar",
    "xtz": "tezos"
}

# Create Quart app
web_app = Quart(__name__)

@web_app.route('/health')
async def health_check():
    return jsonify({"status": "OK"})

# --- Asynchronous HTTP helper using aiohttp ---
async def fetch_json(url, params=None, headers=None):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                logging.error(f"Error fetching {url}: {response.status} {text}")
                raise ValueError(f"Error fetching {url}: {response.status}")
            return await response.json()

# --- Data fetching functions ---
# Historical data from CoinGecko using the static mapping
async def get_crypto_data(ticker: str):
    coin_id = COIN_MAPPING.get(ticker.lower(), ticker.lower())
    cache_key = f"{coin_id}_market_chart"
    # Cache duration in seconds (e.g., 300 seconds = 5 minutes)
    cache_duration = 300
    if cache_key in cache and time() - cache[cache_key]['timestamp'] < cache_duration:
        logging.info("Using cached data")
        return cache[cache_key]['data']
    
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart?vs_currency=usd&days=365"
    data = await fetch_json(url)
    if 'prices' not in data or 'total_volumes' not in data:
        logging.error(f"Invalid data received from CoinGecko for {coin_id}: {data}")
        raise ValueError("Invalid data received from CoinGecko API.")
    
    prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'prices'])
    volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    df = pd.merge(prices_df, volumes_df, on='timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Cache the result along with a timestamp
    cache[cache_key] = {"timestamp": time(), "data": df}
    return df

# Real-time data from Binance (wrapped synchronous call)
async def get_real_time_data(ticker: str):
    def fetch():
        ticker_data = BINANCE_API.fetch_ticker(f"{ticker.upper()}/USDT")
        return {
            'price': ticker_data['last'],
            'volume': ticker_data['baseVolume'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    return await asyncio.to_thread(fetch)

# Sentiment analysis from Twitter using aiohttp
async def get_sentiment(ticker: str):
    query = f"{ticker} cryptocurrency"
    params = {'query': query, 'max_results': 100}
    headers = {"Authorization": f"Bearer {TWITTER_BEARER}"}
    data = await fetch_json(SENTIMENT_API, params=params, headers=headers)
    tweets = data.get('data', [])
    sentiments = [TextBlob(tweet.get('text', '')).sentiment.polarity for tweet in tweets]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    return {
        'sentiment': avg_sentiment,
        'tweet_count': len(tweets)
    }

# --- Calculation functions (synchronous) ---
def calculate_volatility(data):
    returns = data['prices'].pct_change().dropna()
    return returns.std() * np.sqrt(365)

def calculate_risk_reward(data, prediction):
    current_price = data['prices'].iloc[-1]
    stop_loss = current_price * 0.95  # 5% stop loss
    take_profit = prediction * 1.10   # 10% target
    risk = current_price - stop_loss
    if risk == 0:
        risk = 1e-6
    reward = take_profit - current_price
    return reward / risk

# --- Prediction strategy functions (synchronous) ---
def technical_analysis_prediction(data):
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

# --- Telegram Bot Handlers ---
async def predict_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ticker = context.args[0].lower() if context.args else 'btc'
    try:
        # Asynchronously fetch historical data, real-time data, and sentiment
        crypto_data = await get_crypto_data(ticker)
        real_time_data = await get_real_time_data(ticker)
        sentiment = await get_sentiment(ticker)
        
        # Run prediction strategies synchronously (could be offloaded if needed)
        tech_result = technical_analysis_prediction(crypto_data)
        lstm_result = lstm_prediction(crypto_data)
        prophet_result = prophet_prediction(crypto_data)
        results = [tech_result, lstm_result, prophet_result]
        best_strategy = max(results, key=lambda x: x['confidence'])
        risk_reward = calculate_risk_reward(crypto_data, best_strategy['prediction'])
        
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
            if 'signals' in result and result['signals']:
                response += f"  Signals: {', '.join(result['signals'])}\n"
        response += (
            f"\n🌟 **Recommended Strategy**: {best_strategy['strategy']} "
            f"(Confidence: {best_strategy['confidence']*100:.1f}%)\n"
            f"📊 **Risk/Reward Ratio**: {risk_reward:.2f}\n"
        )
        await update.message.reply_text(response)
    except Exception as e:
        logging.error(f"Error in predict_handler: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Crypto Prediction Bot\nUse /predict [ticker]")



async def main():
    # Start Quart server in the background
    quart_task = asyncio.create_task(web_app.run_task(
        host='0.0.0.0', 
        port=10000,
        use_reloader=False
    ))
    
    # Start Telegram bot
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("predict", predict_handler))
    
    # Run the bot
    await application.run_polling()

if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            asyncio.run(main())
    except Exception as e:
        logging.error(f"Error starting bot: {e}")