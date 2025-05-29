
import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import logging

class CircularBuffer:
    """Memory-efficient circular buffer for time series data"""
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.full(size, np.nan)
        self.index = 0
        self.count = 0
        self.lock = threading.Lock()
    
    def add(self, value: float):
        """Add a value to the buffer"""
        with self.lock:
            self.buffer[self.index] = value
            self.index = (self.index + 1) % self.size
            self.count = min(self.count + 1, self.size)
    
    def get_array(self) -> np.ndarray:
        """Get the buffer as a properly ordered numpy array"""
        with self.lock:
            if self.count < self.size:
                # Buffer not full yet
                return self.buffer[:self.count].copy()
            else:
                # Buffer is full, need to reorder
                return np.concatenate([
                    self.buffer[self.index:],
                    self.buffer[:self.index]
                ])
    
    def get_latest(self, n: int = 1) -> np.ndarray:
        """Get the n most recent values"""
        with self.lock:
            if n >= self.count:
                return self.get_array()
            
            if self.count < self.size:
                return self.buffer[max(0, self.count - n):self.count].copy()
            else:
                start_idx = (self.index - n) % self.size
                if start_idx < self.index:
                    return self.buffer[start_idx:self.index].copy()
                else:
                    return np.concatenate([
                        self.buffer[start_idx:],
                        self.buffer[:self.index]
                    ])
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.count >= self.size
    
    def __len__(self) -> int:
        return self.count

class StockDataCache:
    """Cache for individual stock data"""
    
    def __init__(self, symbol: str, max_prices: int = 200, max_volumes: int = 50):
        self.symbol = symbol
        self.prices = CircularBuffer(max_prices)
        self.volumes = CircularBuffer(max_volumes)
        self.timestamps = deque(maxlen=max_prices)
        
        # RSI calculation components
        self.gains = CircularBuffer(14)  # Default RSI period
        self.losses = CircularBuffer(14)
        self.last_price = None
        self.current_rsi = None
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        
        # Additional indicators cache
        self.ema_12 = None
        self.ema_26 = None
        self.sma_20 = None
        self.sma_50 = None
        
        self.lock = threading.RLock()
    
    def add_price_data(self, price: float, volume: int, timestamp: datetime):
        """Add new price and volume data"""
        with self.lock:
            # Add to buffers
            self.prices.add(price)
            self.volumes.add(volume)
            self.timestamps.append(timestamp)
            
            # Calculate gain/loss for RSI
            if self.last_price is not None:
                change = price - self.last_price
                if change > 0:
                    self.gains.add(change)
                    self.losses.add(0.0)
                else:
                    self.gains.add(0.0)
                    self.losses.add(abs(change))
                
                # Update RSI if we have enough data
                self._update_rsi()
            
            self.last_price = price
            
            # Update moving averages
            self._update_moving_averages()
    
    def _update_rsi(self):
        """Update RSI calculation incrementally"""
        gains_array = self.gains.get_array()
        losses_array = self.losses.get_array()
        
        if len(gains_array) < 14:
            return
        
        # Use Wilder's smoothing method for RSI
        if self.avg_gain == 0.0 and self.avg_loss == 0.0:
            # First calculation - use simple average
            self.avg_gain = np.mean(gains_array)
            self.avg_loss = np.mean(losses_array)
        else:
            # Incremental update
            latest_gain = gains_array[-1] if len(gains_array) > 0 else 0
            latest_loss = losses_array[-1] if len(losses_array) > 0 else 0
            
            self.avg_gain = ((self.avg_gain * 13) + latest_gain) / 14
            self.avg_loss = ((self.avg_loss * 13) + latest_loss) / 14
        
        # Calculate RSI
        if self.avg_loss == 0:
            self.current_rsi = 100
        else:
            rs = self.avg_gain / self.avg_loss
            self.current_rsi = 100 - (100 / (1 + rs))
    
    def _update_moving_averages(self):
        """Update exponential and simple moving averages"""
        prices_array = self.prices.get_array()
        
        if len(prices_array) == 0:
            return
        
        current_price = prices_array[-1]
        
        # EMA calculations (using standard multipliers)
        if self.ema_12 is None:
            self.ema_12 = current_price
        else:
            multiplier_12 = 2 / (12 + 1)
            self.ema_12 = (current_price * multiplier_12) + (self.ema_12 * (1 - multiplier_12))
        
        if self.ema_26 is None:
            self.ema_26 = current_price
        else:
            multiplier_26 = 2 / (26 + 1)
            self.ema_26 = (current_price * multiplier_26) + (self.ema_26 * (1 - multiplier_26))
        
        # SMA calculations
        if len(prices_array) >= 20:
            self.sma_20 = np.mean(prices_array[-20:])
        
        if len(prices_array) >= 50:
            self.sma_50 = np.mean(prices_array[-50:])
    
    def get_rsi(self) -> Optional[float]:
        """Get current RSI value"""
        return self.current_rsi
    
    def get_latest_price(self) -> Optional[float]:
        """Get the most recent price"""
        if len(self.prices) > 0:
            return self.prices.get_latest(1)[0]
        return None
    
    def get_price_history(self, periods: int = None) -> np.ndarray:
        """Get price history"""
        if periods is None:
            return self.prices.get_array()
        return self.prices.get_latest(periods)
    
    def get_indicators(self) -> Dict:
        """Get all current indicator values"""
        return {
            'rsi': self.current_rsi,
            'ema_12': self.ema_12,
            'ema_26': self.ema_26,
            'sma_20': self.sma_20,
            'sma_50': self.sma_50,
            'latest_price': self.get_latest_price(),
            'avg_gain': self.avg_gain,
            'avg_loss': self.avg_loss
        }

class CacheManager:
    """Main cache manager for all stock data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stock_caches = {}
        self.symbols = config.get('symbols', [])
        self.logger = logging.getLogger(__name__)
        
        # Initialize caches for all symbols
        for symbol in self.symbols:
            self.stock_caches[symbol] = StockDataCache(symbol)
        
        # System metrics
        self.update_count = 0
        self.start_time = datetime.now()
        
        self.logger.info(f"Cache manager initialized for {len(self.symbols)} symbols")
    
    async def update_price_data(self, symbol: str, data: Dict):
        """Update price data for a symbol (callback from market data feed)"""
        if symbol not in self.stock_caches:
            self.stock_caches[symbol] = StockDataCache(symbol)
        
        cache = self.stock_caches[symbol]
        cache.add_price_data(
            price=data['price'],
            volume=data['volume'],
            timestamp=data['timestamp']
        )
        
        self.update_count += 1
        
        # Log periodic updates
        if self.update_count % 100 == 0:
            self.logger.info(f"Processed {self.update_count} price updates")
    
    def get_stock_cache(self, symbol: str) -> Optional[StockDataCache]:
        """Get cache for a specific symbol"""
        return self.stock_caches.get(symbol)
    
    def get_all_rsi_values(self) -> Dict[str, float]:
        """Get current RSI values for all symbols"""
        rsi_values = {}
        for symbol, cache in self.stock_caches.items():
            rsi = cache.get_rsi()
            if rsi is not None:
                rsi_values[symbol] = rsi
        return rsi_values
    
    def get_all_indicators(self) -> Dict[str, Dict]:
        """Get all indicators for all symbols"""
        indicators = {}
        for symbol, cache in self.stock_caches.items():
            indicators[symbol] = cache.get_indicators()
        return indicators
    
    def get_ready_symbols(self) -> List[str]:
        """Get symbols that have enough data for RSI calculation"""
        ready = []
        for symbol, cache in self.stock_caches.items():
            if cache.get_rsi() is not None:
                ready.append(symbol)
        return ready
    
    def get_system_stats(self) -> Dict:
        """Get cache system statistics"""
        uptime = datetime.now() - self.start_time
        ready_count = len(self.get_ready_symbols())
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'total_updates': self.update_count,
            'symbols_tracked': len(self.stock_caches),
            'symbols_ready': ready_count,
            'updates_per_minute': (self.update_count / uptime.total_seconds()) * 60 if uptime.total_seconds() > 0 else 0
        }
    
    def clear_cache(self, symbol: str = None):
        """Clear cache for specific symbol or all symbols"""
        if symbol:
            if symbol in self.stock_caches:
                del self.stock_caches[symbol]
                self.logger.info(f"Cleared cache for {symbol}")
        else:
            self.stock_caches.clear()
            self.update_count = 0
            self.start_time = datetime.now()
            self.logger.info("Cleared all caches")

# Example usage and testing
def test_cache_manager():
    """Test the cache manager functionality"""
    import asyncio
    
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT']
    }
    
    cache_manager = CacheManager(config)
    
    # Simulate price updates
    async def simulate_data():
        import random
        base_prices = {'AAPL': 150, 'GOOGL': 2500, 'MSFT': 300}
        
        for i in range(50):  # Simulate 50 updates
            for symbol in config['symbols']:
                # Generate realistic price movement
                base_price = base_prices[symbol]
                price_change = random.uniform(-0.02, 0.02)  # ±2% change
                new_price = base_price * (1 + price_change)
                base_prices[symbol] = new_price
                
                data = {
                    'price': new_price,
                    'volume': random.randint(1000, 10000),
                    'timestamp': datetime.now()
                }
                
                await cache_manager.update_price_data(symbol, data)
            
            await asyncio.sleep(0.1)  # 100ms between updates
        
        # Print results
        print("\nFinal RSI Values:")
        rsi_values = cache_manager.get_all_rsi_values()
        for symbol, rsi in rsi_values.items():
            print(f"{symbol}: RSI = {rsi:.2f}")
        
        print(f"\nSystem Stats:")
        stats = cache_manager.get_system_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    # Run the test
    asyncio.run(simulate_data())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_cache_manager()