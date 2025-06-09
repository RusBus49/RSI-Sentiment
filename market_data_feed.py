"""
Market Data Feed Manager - Yahoo Finance Only
Simplified, reliable market data feed using only Yahoo Finance
Optimized for RSI trading with proper rate limiting and error handling
"""

import asyncio
import aiohttp
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

@dataclass
class RateLimiter:
    """Rate limiter specifically for Yahoo Finance API limits"""
    requests_per_minute: int = 50  # Conservative vs 60 limit
    requests_per_hour: int = 300   # Conservative vs 360 limit
    requests_per_day: int = 7000   # Conservative vs 8000 limit
    
    def __post_init__(self):
        self.minute_requests = []
        self.hour_requests = []
        self.day_requests = []
    
    def can_make_request(self) -> bool:
        """Check if we can make a request without exceeding limits"""
        now = time.time()
        self._clean_old_requests(now)
        
        return (len(self.minute_requests) < self.requests_per_minute and
                len(self.hour_requests) < self.requests_per_hour and
                len(self.day_requests) < self.requests_per_day)
    
    def record_request(self):
        """Record a request timestamp"""
        now = time.time()
        self.minute_requests.append(now)
        self.hour_requests.append(now)
        self.day_requests.append(now)
    
    def time_until_next_request(self) -> float:
        """Get seconds to wait until next request is allowed"""
        if self.can_make_request():
            return 0
        
        now = time.time()
        self._clean_old_requests(now)
        
        wait_times = []
        
        if len(self.minute_requests) >= self.requests_per_minute:
            oldest_minute = min(self.minute_requests)
            wait_times.append(60 - (now - oldest_minute))
        
        if len(self.hour_requests) >= self.requests_per_hour:
            oldest_hour = min(self.hour_requests)
            wait_times.append(3600 - (now - oldest_hour))
        
        if len(self.day_requests) >= self.requests_per_day:
            oldest_day = min(self.day_requests)
            wait_times.append(86400 - (now - oldest_day))
        
        return max(wait_times) if wait_times else 1.0
    
    def _clean_old_requests(self, now: float):
        """Remove old request timestamps"""
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400
        
        self.minute_requests = [t for t in self.minute_requests if t > minute_ago]
        self.hour_requests = [t for t in self.hour_requests if t > hour_ago]
        self.day_requests = [t for t in self.day_requests if t > day_ago]
    
    def get_stats(self) -> Dict:
        """Get current rate limiting statistics"""
        now = time.time()
        self._clean_old_requests(now)
        
        return {
            'requests_last_minute': len(self.minute_requests),
            'requests_last_hour': len(self.hour_requests),
            'requests_today': len(self.day_requests),
            'can_make_request': self.can_make_request(),
            'wait_time_seconds': self.time_until_next_request()
        }

class MarketDataFeed:
    """Simplified market data feed using only Yahoo Finance"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config.get('symbols', [])
        self.update_interval = max(60, config.get('update_interval', 60))  # Minimum 60 seconds
        
        # Rate limiter
        self.rate_limiter = RateLimiter()
        
        # Session and callbacks
        self.session = None
        self.callbacks = []
        self.is_running = False
        self.latest_prices = {}
        
        # Error tracking
        self.error_counts = {}
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rate_limited_requests = 0
        self.start_time = None
        
        # Initialize error tracking
        for symbol in self.symbols:
            self.error_counts[symbol] = 0
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the market data feed"""
        self.is_running = True
        self.start_time = datetime.now()
        self.consecutive_failures = 0
        
        # Create session with Yahoo Finance optimized headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        timeout = aiohttp.ClientTimeout(total=15)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=aiohttp.TCPConnector(limit=2, limit_per_host=1)
        )
        
        self.logger.info(f"Market data feed started for {len(self.symbols)} symbols")
        self.logger.info(f"Update interval: {self.update_interval} seconds")
        
        await self._run_data_loop()
    
    async def stop(self):
        """Stop the market data feed"""
        self.is_running = False
        
        if self.session:
            await self.session.close()
        
        # Log final statistics
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        rate_stats = self.rate_limiter.get_stats()
        
        self.logger.info("Market data feed stopped")
        self.logger.info(f"Uptime: {uptime:.0f} seconds")
        self.logger.info(f"Total requests: {self.total_requests}")
        self.logger.info(f"Success rate: {(self.successful_requests/max(1,self.total_requests)*100):.1f}%")
        self.logger.info(f"Requests today: {rate_stats['requests_today']}")
    
    def add_callback(self, callback: Callable):
        """Add callback function to receive price updates"""
        self.callbacks.append(callback)
    
    async def _run_data_loop(self):
        """Main loop for collecting market data with rate limiting"""
        while self.is_running:
            cycle_start = datetime.now()
            successful_updates = 0
            
            self.logger.debug(f"Starting data cycle for {len(self.symbols)} symbols")
            
            # Process symbols sequentially with rate limiting
            for i, symbol in enumerate(self.symbols):
                if not self.is_running:
                    break
                
                try:
                    data = await self._fetch_symbol_data(symbol)
                    if data:
                        successful_updates += 1
                        self.error_counts[symbol] = 0  # Reset error count on success
                        self.consecutive_failures = 0
                        
                        # Cache the data
                        self.latest_prices[symbol] = data
                        
                        # Notify callbacks
                        for callback in self.callbacks:
                            try:
                                await callback(symbol, data)
                            except Exception as e:
                                self.logger.error(f"Callback error for {symbol}: {e}")
                    else:
                        self.error_counts[symbol] += 1
                        self.consecutive_failures += 1
                        
                        # Log errors for first few attempts
                        if self.error_counts[symbol] <= 3:
                            self.logger.warning(f"Failed to get data for {symbol} (attempt {self.error_counts[symbol]})")
                
                except Exception as e:
                    self.error_counts[symbol] += 1
                    self.consecutive_failures += 1
                    self.logger.error(f"Error processing {symbol}: {e}")
                
                # Rate limiting delay between symbols
                if i < len(self.symbols) - 1:
                    # Minimum 1.5 second delay between requests
                    delay = max(1.5, 60 / self.rate_limiter.requests_per_minute)
                    await asyncio.sleep(delay)
            
            # Check for too many consecutive failures
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.logger.error(f"Too many consecutive failures ({self.consecutive_failures}). Pausing for 5 minutes.")
                await asyncio.sleep(300)  # 5 minute pause
                self.consecutive_failures = 0
            
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            
            # Log cycle summary
            if successful_updates > 0:
                self.logger.info(f"Data cycle complete: {successful_updates}/{len(self.symbols)} symbols in {cycle_time:.1f}s")
            else:
                self.logger.warning(f"Data cycle failed: 0/{len(self.symbols)} symbols successful")
            
            # Wait for next cycle
            remaining_time = max(10, self.update_interval - cycle_time)
            await asyncio.sleep(remaining_time)
    
    async def _fetch_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data for a single symbol with comprehensive error handling"""
        # Check rate limits
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.time_until_next_request()
            self.logger.debug(f"Rate limited - waiting {wait_time:.1f}s for {symbol}")
            self.rate_limited_requests += 1
            await asyncio.sleep(wait_time)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        
        try:
            self.total_requests += 1
            self.rate_limiter.record_request()
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse the Yahoo Finance response
                    chart = data.get('chart', {})
                    if not chart:
                        self.failed_requests += 1
                        return None
                    
                    result = chart.get('result', [])
                    if not result:
                        self.failed_requests += 1
                        return None
                    
                    chart_data = result[0]
                    meta = chart_data.get('meta', {})
                    
                    if not meta:
                        self.failed_requests += 1
                        return None
                    
                    # Extract price and volume data
                    price = meta.get('regularMarketPrice') or meta.get('previousClose')
                    volume = meta.get('regularMarketVolume', 0)
                    market_state = meta.get('marketState', 'UNKNOWN')
                    
                    if price is None:
                        self.failed_requests += 1
                        self.logger.debug(f"No price data for {symbol}")
                        return None
                    
                    self.successful_requests += 1
                    
                    return {
                        'symbol': symbol,
                        'price': float(price),
                        'volume': int(volume) if volume else 0,
                        'market_state': market_state,
                        'timestamp': datetime.now(),
                        'source': 'yahoo_finance'
                    }
                
                elif response.status == 429:
                    # Rate limited by Yahoo
                    self.rate_limited_requests += 1
                    retry_after = int(response.headers.get('Retry-After', '60'))
                    self.logger.warning(f"Rate limited by Yahoo for {symbol}. Waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return None
                
                elif response.status == 404:
                    # Symbol not found
                    self.failed_requests += 1
                    self.logger.warning(f"Symbol {symbol} not found (404)")
                    return None
                
                else:
                    # Other HTTP error
                    self.failed_requests += 1
                    self.logger.warning(f"HTTP {response.status} for {symbol}")
                    return None
        
        except asyncio.TimeoutError:
            self.failed_requests += 1
            self.logger.warning(f"Timeout fetching {symbol}")
            return None
        
        except aiohttp.ClientError as e:
            self.failed_requests += 1
            self.logger.warning(f"Client error for {symbol}: {e}")
            return None
        
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"Unexpected error fetching {symbol}: {e}")
            return None
    
    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get the latest cached price for a symbol"""
        return self.latest_prices.get(symbol)
    
    def get_all_latest_prices(self) -> Dict:
        """Get all latest cached prices"""
        return self.latest_prices.copy()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive feed statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        rate_stats = self.rate_limiter.get_stats()
        
        return {
            'uptime_seconds': uptime,
            'symbols_configured': len(self.symbols),
            'symbols_with_data': len(self.latest_prices),
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'rate_limited_requests': self.rate_limited_requests,
            'success_rate_percent': (self.successful_requests / max(1, self.total_requests)) * 100,
            'consecutive_failures': self.consecutive_failures,
            'update_interval': self.update_interval,
            **rate_stats
        }
    
    def get_error_summary(self) -> Dict:
        """Get error summary by symbol"""
        return self.error_counts.copy()

# Example usage and testing
async def test_yahoo_feed():
    """Test the simplified Yahoo Finance feed"""
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'update_interval': 70  # 70 seconds between cycles
    }
    
    feed = MarketDataFeed(config)
    
    # Add callback to print updates
    async def print_update(symbol, data):
        market_emoji = "🟢" if data['market_state'] == 'REGULAR' else "🟡"
        print(f"{market_emoji} {symbol}: ${data['price']:.2f} "
              f"(Vol: {data['volume']:,}) [{data['market_state']}] "
              f"at {data['timestamp'].strftime('%H:%M:%S')}")
    
    feed.add_callback(print_update)
    
    # Add periodic stats logging
    async def log_stats():
        while feed.is_running:
            await asyncio.sleep(180)  # Every 3 minutes
            stats = feed.get_statistics()
            print(f"\n📊 Feed Stats: {stats['success_rate_percent']:.1f}% success, "
                  f"{stats['requests_today']} requests today, "
                  f"can request: {stats['can_make_request']}")
    
    try:
        print("🚀 Starting Yahoo Finance Feed Test")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        # Run both the feed and stats logging
        await asyncio.gather(
            feed.start(),
            log_stats()
        )
        
    except KeyboardInterrupt:
        print("\n👋 Stopping feed...")
    finally:
        await feed.stop()

if __name__ == "__main__":
    asyncio.run(test_yahoo_feed())