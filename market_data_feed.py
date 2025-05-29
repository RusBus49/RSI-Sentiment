
import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import time

class MarketDataFeed:
    def __init__(self, config: Dict):
        self.config = config
        self.session = None
        self.callbacks = []
        self.is_running = False
        self.symbols = config.get('symbols', [])
        self.update_interval = config.get('update_interval', 5)  # seconds
        self.api_key = config.get('api_key', '')
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Data cache for latest prices
        self.latest_prices = {}
        
    async def start(self):
        """Start the market data feed"""
        self.is_running = True
        self.session = aiohttp.ClientSession()
        self.logger.info("Market data feed started")
        
        # Start the main data collection loop
        await self._run_data_loop()
    
    async def stop(self):
        """Stop the market data feed"""
        self.is_running = False
        if self.session:
            await self.session.close()
        self.logger.info("Market data feed stopped")
    
    def add_callback(self, callback: Callable):
        """Add callback function to receive price updates"""
        self.callbacks.append(callback)
    
    async def _run_data_loop(self):
        """Main loop for collecting market data"""
        while self.is_running:
            try:
                # Get data for all symbols
                tasks = [self._fetch_symbol_data(symbol) for symbol in self.symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for symbol, result in zip(self.symbols, results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Error fetching data for {symbol}: {result}")
                        continue
                    
                    if result:
                        # Update cache
                        self.latest_prices[symbol] = result
                        
                        # Notify callbacks
                        for callback in self.callbacks:
                            try:
                                await callback(symbol, result)
                            except Exception as e:
                                self.logger.error(f"Callback error: {e}")
                
                # Wait before next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Data loop error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _fetch_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data for a single symbol with rate limiting"""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
        
        try:
            # Try Alpha Vantage first (primary source)
            data = await self._fetch_alpha_vantage(symbol)
            if data:
                return data
            
            # Fallback to Yahoo Finance
            data = await self._fetch_yahoo_finance(symbol)
            if data:
                return data
                
            self.logger.warning(f"No data available for {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    async def _fetch_alpha_vantage(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Alpha Vantage API"""
        if not self.api_key:
            return None
            
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    quote = data.get('Global Quote', {})
                    
                    if quote:
                        return {
                            'symbol': symbol,
                            'price': float(quote.get('05. price', 0)),
                            'volume': int(float(quote.get('06. volume', 0))),
                            'timestamp': datetime.now(),
                            'source': 'alpha_vantage'
                        }
        except Exception as e:
            self.logger.debug(f"Alpha Vantage error for {symbol}: {e}")
        
        return None
    
    async def _fetch_yahoo_finance(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Yahoo Finance (free fallback)"""
        # Note: This is a simplified implementation
        # In production, you'd want to use yfinance library or similar
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get('chart', {}).get('result', [])
                    
                    if result:
                        chart_data = result[0]
                        meta = chart_data.get('meta', {})
                        
                        return {
                            'symbol': symbol,
                            'price': meta.get('regularMarketPrice', 0),
                            'volume': meta.get('regularMarketVolume', 0),
                            'timestamp': datetime.now(),
                            'source': 'yahoo_finance'
                        }
        except Exception as e:
            self.logger.debug(f"Yahoo Finance error for {symbol}: {e}")
        
        return None
    
    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get the latest cached price for a symbol"""
        return self.latest_prices.get(symbol)
    
    def get_all_latest_prices(self) -> Dict:
        """Get all latest cached prices"""
        return self.latest_prices.copy()

# Example usage and testing
async def main():
    """Test the market data feed"""
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'update_interval': 10,
        'api_key': ''  # Add your Alpha Vantage API key here
    }
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create feed
    feed = MarketDataFeed(config)
    
    # Add callback to print updates
    async def price_callback(symbol, data):
        print(f"{symbol}: ${data['price']:.2f} at {data['timestamp']}")
    
    feed.add_callback(price_callback)
    
    try:
        # Run for 60 seconds
        await asyncio.wait_for(feed.start(), timeout=60)
    except asyncio.TimeoutError:
        print("Test completed")
    finally:
        await feed.stop()

if __name__ == "__main__":
    asyncio.run(main())