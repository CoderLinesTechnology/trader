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

# Apply nest_asyncio for async loop compatibility
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Validate environment variables
REQUIRED_ENV = ['TELEGRAM_TOKEN', 'TWITTER_BEARER']
missing = [var for var in REQUIRED_ENV if not os.getenv(var)]
if missing:
    logger.critical(f"Missing environment variables: {', '.join(missing)}")
    exit(1)

# Configuration
CONFIG = {
    'COINGECKO_API': "https://api.coingecko.com/api/v3",
    'DAYS_TO_PREDICT': 7,
    'CACHE_TTL': 300,
    'TWITTER_RATE_LIMIT': 60,
    'PORT': 10000
}

# Initialize components
cache = {}
binance_api = ccxt.binance()
web_app = Quart(__name__)

class TwitterAPI:
    """Managed Twitter API client with async session handling"""
    def __init__(self):
        self.session = None
        self.last_request = 0

    async def __aenter__(self):
        await self.start_session()
        return self

    async def __aexit__(self, *exc):
        await self.close_session()

    async def start_session(self):
        """Initialize async session"""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Close async session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_sentiment(self, ticker: str):
        """Get crypto sentiment with rate limiting"""
        try:
            await self.start_session()
            elapsed = time() - self.last_request
            if elapsed < CONFIG['TWITTER_RATE_LIMIT']:
                await asyncio.sleep(CONFIG['TWITTER_RATE_LIMIT'] - elapsed)

            params = {
                'query': f'{ticker} cryptocurrency',
                'max_results': 50,
                'tweet.fields': 'created_at'
            }
            headers = {'Authorization': f'Bearer {os.getenv("TWITTER_BEARER")}'}

            async with self.session.get(
                'https://api.twitter.com/2/tweets/search/recent',
                params=params,
                headers=headers
            ) as response:
                self.last_request = time()
                if response.status == 429:
                    logger.warning("Twitter API rate limit reached")
                    return {'sentiment': 0, 'count': 0}
                
                if response.status != 200:
                    logger.error(f"Twitter API error: {response.status}")
                    return {'sentiment': 0, 'count': 0}

                data = await response.json()
                tweets = data.get('data', [])
                sentiments = [
                    TextBlob(t['text']).sentiment.polarity 
                    for t in tweets if 'text' in t
                ]
                return {
                    'sentiment': np.mean(sentiments) if sentiments else 0,
                    'count': len(tweets)
                }

        except Exception as e:
            logger.error(f"Twitter API failure: {str(e)}")
            return {'sentiment': 0, 'count': 0}

# Crypto data management
COIN_MAPPING = {
    'btc': 'bitcoin', 'eth': 'ethereum', 'doge': 'dogecoin',
    'bnb': 'binancecoin', 'ada': 'cardano', 'xrp': 'ripple',
    'sol': 'solana', 'dot': 'polkadot', 'matic': 'matic-network'
}

async def fetch_data(url: str):
    """Generic API fetcher with error handling"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"API Error {response.status} from {url}")
                    return None
                return await response.json()
    except Exception as e:
        logger.error(f"Fetch error: {str(e)}")
        return None

async def get_crypto_data(ticker: str):
    """Get historical data with caching"""
    cache_key = f'coingecko_{ticker}'
    coin_id = COIN_MAPPING.get(ticker.lower(), ticker.lower())
    
    if cache_key in cache:
        if time() - cache[cache_key]['timestamp'] < CONFIG['CACHE_TTL']:
            logger.info("Using cached CoinGecko data")
            return cache[cache_key]['data']
    
    url = f"{CONFIG['COINGECKO_API']}/coins/{coin_id}/market_chart?vs_currency=usd&days=365"
    
    data = await fetch_data(f"{url}")
    if not data or 'prices' not in data:
        return None
    
    try:
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        cache[cache_key] = {'timestamp': time(), 'data': df}
        return df
    except Exception as e:
        logger.error(f"Data processing error: {str(e)}")
        return None

# Prediction models
def calculate_volatility(data: pd.DataFrame):
    """Calculate annualized volatility"""
    returns = data['price'].pct_change().dropna()
    return returns.std() * np.sqrt(365)

def technical_analysis(df: pd.DataFrame):
    """Technical analysis prediction"""
    df = df[-200:]  # Use last 200 data points
    df['ma50'] = df['price'].rolling(50).mean()
    df['ma200'] = df['price'].rolling(200).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['price'], window=14).rsi()
    
    signals = []
    if df['ma50'].iloc[-1] > df['ma200'].iloc[-1]:
        signals.append('Golden Cross')
    if df['rsi'].iloc[-1] < 30:
        signals.append('Oversold')
    elif df['rsi'].iloc[-1] > 70:
        signals.append('Overbought')
    
    prediction = df['price'].iloc[-1] * (1 + 0.005 * len(signals))
    volatility = calculate_volatility(df)
    confidence = min(0.9, 0.3 + 0.1 * len(signals)) * (1 - volatility)
    
    return {
        'price': prediction,
        'confidence': confidence,
        'signals': signals
    }

def lstm_prediction(df: pd.DataFrame):
    """LSTM model prediction"""
    data = df['price'].values[-60:]
    if len(data) < 60:
        return {'price': None, 'error': 'Insufficient data'}
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(scaled_data, scaled_data, epochs=10, batch_size=32, verbose=0)
    
    future = []
    current_batch = scaled_data[-60:].reshape(1, 60, 1)
    for _ in range(CONFIG['DAYS_TO_PREDICT']):
        pred = model.predict(current_batch)[0, 0]
        future.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)
    
    return {
        'price': scaler.inverse_transform([future])[0][-1],
        'confidence': 0.7 * (1 - calculate_volatility(df))
    }

def prophet_prediction(df: pd.DataFrame):
    """Prophet model prediction"""
    try:
        prophet_df = df.reset_index().rename(columns={'timestamp': 'ds', 'price': 'y'})
        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=CONFIG['DAYS_TO_PREDICT'])
        forecast = model.predict(future)
        return {
            'price': forecast['yhat'].iloc[-1],
            'confidence': 0.65 * (1 - calculate_volatility(df))
        }
    except Exception as e:
        logger.error(f"Prophet error: {str(e)}")
        return {'price': None, 'error': str(e)}

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_markdown(
        "🚀 *Crypto Prediction Bot*\n\n"
        "Supported coins: BTC, ETH, DOGE, BNB, ADA, XRP, SOL, DOT, MATIC\n"
        "Usage: `/predict <ticker>`\n"
        "Example: `/predict btc`"
    )

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle prediction requests"""
    try:
        ticker = context.args[0].lower() if context.args else 'btc'
        cache_key = f"prediction_{ticker}"
        
        # Check cache first
        if cache_key in cache and time() - cache[cache_key]['time'] < CONFIG['CACHE_TTL']:
            logger.info("Using cached prediction")
            results = cache[cache_key]['data']
        else:
            # Fetch fresh data
            df = await get_crypto_data(ticker)
            if df is None:
                raise ValueError("Could not fetch price data")
            
            # Get predictions
            results = {
                'ta': technical_analysis(df),
                'lstm': lstm_prediction(df),
                'prophet': prophet_prediction(df),
                'sentiment': await twitter_client.get_sentiment(ticker)
            }
            cache[cache_key] = {'time': time(), 'data': results}
        
        # Format response
        response = [
            f"📈 *{ticker.upper()} Predictions*",
            f"📊 Sentiment: {results['sentiment']['sentiment']:.2f} ({results['sentiment']['count']} tweets)",
            "\n*Technical Analysis*:",
            f"Price: ${results['ta']['price']:.2f}",
            f"Confidence: {results['ta']['confidence']:.1%}",
            f"Signals: {', '.join(results['ta']['signals'] or ['None'])}",
            "\n*LSTM Model*:",
            f"Price: ${results['lstm']['price']:.2f}",
            f"Confidence: {results['lstm']['confidence']:.1%}",
            "\n*Prophet Model*:",
            f"Price: ${results['prophet']['price']:.2f}",
            f"Confidence: {results['prophet']['confidence']:.1%}",
        ]
        
        await update.message.reply_markdown('\n'.join(response))
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        await update.message.reply_markdown("❌ *Error*: Could not generate predictions. Please try again later.")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Handle Telegram errors"""
    logger.error(f"Update {update} caused error: {context.error}")
    if update and hasattr(update, 'message'):
        await update.message.reply_text("⚠️ An error occurred. Please try again later.")

async def run_web_server():
    """Run Quart web server for Render requirements"""
    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', CONFIG['PORT'])
    await site.start()

@web_app.route('/health')
async def health_check():
    return jsonify(status="OK", timestamp=datetime.utcnow().isoformat())

async def main():
    """Main async entry point with proper resource management"""
    async with TwitterAPI() as twitter_client:
        # Create Telegram application
        application = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()
        
        # Register handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("predict", predict))
        application.add_error_handler(error_handler)
        
        # Start services
        await asyncio.gather(
            run_web_server(),
            application.run_polling()
        )

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Critical failure: {str(e)}")