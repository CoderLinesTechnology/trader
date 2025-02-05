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
    
    url = f"{CONFIG['COINGECKO_API']}/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': 365}
    
    data = await fetch_data(f"{url}?{aiohttp.client.helpers.urlencode(params)}")
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

# Prediction models (keep the same as before)
# ... [Keep all prediction model functions unchanged] ...

# Telegram bot handlers (keep the same as before)
# ... [Keep all Telegram handler functions unchanged] ...

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